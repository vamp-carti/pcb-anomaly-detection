#include <atomic>
#include <chrono>
#include <cstdlib> 
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>
#include <condition_variable>

// Hardware and Framework Stack
#include <open62541.h>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <opencv2/opencv.hpp> 
#include <opencv2/cudawarping.hpp> 
#include <opencv2/cudaarithm.hpp> 
#include <opencv2/cudaimgproc.hpp> 
#include <opencv2/cudafilters.hpp> 
#include <opencv2/core/cuda_stream_accessor.hpp> 

namespace fs = std::filesystem;

// --- CONFIGURATION CONSTANTS --- 
constexpr int INPUT_H = 256; 
constexpr int INPUT_W = 256; 
constexpr int INPUT_C = 3; 
constexpr size_t INPUT_SIZE = INPUT_H * INPUT_W * INPUT_C * sizeof(float); 
constexpr size_t OUTPUT_SIZE = INPUT_H * INPUT_W * sizeof(float); 
constexpr int RING_BUFFER_SIZE = 10; 

// --- FIXED REGIONAL AND ALIGNMENT TUNING CONSTANTS (ZONAL GUARDS) --- 
constexpr int X_START = 30; 
constexpr int X_END = 90; 
constexpr int Y_START = 95; 
constexpr int Y_END = 165; 
constexpr float AMP_TRIGGER_THRESHOLD = -0.4500f; 
constexpr float PIN_ZONE_BIAS = 0.09f; 

constexpr int OSC_X_START = 155; 
constexpr int OSC_X_END = 180; 
constexpr int OSC_Y_START = 95; 
constexpr int OSC_Y_END = 160; 
constexpr float OSC_BIAS = 0.05f; 

constexpr int VALID_MIN_WIDTH_THRESHOLD = 95;  

// Industrial Handshake / OPC Status Profiles
enum class IndustrialProfile {
    SAFETY = 10,
    BALANCED = 20,
    UPTIME = 30
};

class TRTLogger : public nvinfer1::ILogger { 
    void log(Severity severity, const char* msg) noexcept override { 
        if (severity <= Severity::kWARNING) std::cout << "[TRT] " << msg << std::endl; 
    } 
} gLogger; 

// --- SEPARATED FILE INGESTION STRUCT WITH GROUND TRUTH LABELS --- 
struct FrameData { 
    cv::Mat frame; 
    std::string filename; 
    int label; // 0 = Good, 1 = Anomaly
    double io_read_ms; // Extracted performance tracker item
}; 


// --- PRODUCTION ENVIRONMENT INGESTION HELPERS ---
// Safely pulls text configuration strings from Docker environment variables
std::string get_env_var(const std::string& key, const std::string& default_value) {
    const char* val = std::getenv(key.c_str());
    return (val == nullptr) ? default_value : std::string(val);
}

// Safely pulls integer profile values from Docker environment variables
int get_env_int(const std::string& key, int default_value) {
    const char* val = std::getenv(key.c_str());
    return (val == nullptr) ? default_value : std::atoi(val);
}

// -- INDUSTRIAL HANDSHAKE ATOMIC REGISTERS ---
std::atomic<bool>  g_opc_trigger{false};           
std::atomic<int>   g_opc_result{0};                
std::atomic<float> g_opc_anomaly_score{0.0000f};   
std::atomic<int>   g_opc_heartbeat{0};             
std::atomic<bool>  g_server_running{true};         // Now it's visible globally!
std::atomic<int>   g_opc_profile_selection{20};

// Global telemetry collection metrics for final report generation
#if ENGINE_BENCHMARK_MODE
std::mutex g_metrics_mutex;
std::vector<double> g_io_read_lats;
std::vector<double> g_preprocess_lats;
std::vector<double> g_h2d_lats;
std::vector<double> g_inference_lats;
std::vector<double> g_postprocess_lats;
std::vector<double> g_d2h_lats;
#endif

// --- EXTERN COMPILATION INTERFACES --- 
extern "C" void launch_hwc_to_chw_interface(const uint8_t* src_data, float* dst_device,  
                                            int width, int height, int src_step,  
                                            cudaStream_t stream); 

extern "C" void launch_postprocess_mask_amp(const float* src_device, float* dst_device, 
                                            int width, int height, 
                                            int final_left, int final_right, 
                                            int y_start, int y_end, 
                                            int x_start, int x_end, 
                                            int osc_y_start, int osc_y_end, 
                                            int osc_x_start, int osc_x_end, 
                                            float amp_trigger_threshold, 
                                            float pin_zone_bias, float osc_bias, 
                                            cudaStream_t stream); 


// Helper function to extract current Host RAM Resident Set Size (RSS) in MegaBytes
double get_host_ram_usage_mb() {
    std::ifstream statm_file("/proc/self/statm");
    if (!statm_file.is_open()) return 0.0;
    
    size_t pages = 0;
    statm_file >> pages; // First value in statm is the total virtual memory size, second is RSS
    statm_file >> pages; // Now pages contains the actual physics RAM page count
    
    // Standard Linux page size is 4KB (4096 bytes)
    double ram_mb = (static_cast<double>(pages) * 4096.0) / (1024.0 * 1024.0);
    return ram_mb;
}

std::string get_iso_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%dT%H:%M:%S");
    return ss.str();
}

// --- TELEMETRY BATCH WRITER LOGGER CLASS ---
class TelemetryLogger {
private:
    std::string log_path_;
    std::vector<std::string> buffer_;
    size_t batch_threshold_ = 50;

public:
    TelemetryLogger(const std::string& path) : log_path_(path) {
        buffer_.reserve(batch_threshold_);
    }

    void log_event(const std::string& board_id, const std::string& status, const std::string& profile,
                   double h2d, double exec, double d2h, double vram_free) {
        std::stringstream ss;
        ss << "{\"timestamp\":\"" << get_iso_timestamp() << "\","
           << "\"board_id\":\"" << board_id << "\","
           << "\"status\":\"" << status << "\",";
        
        // If it's a mechanical error, inject the ME01 tracking code into the json record
        if (status == "MECHANICAL_ERROR") {
            ss << "\"error_code\":\"ME01\",";
        }

        ss << "\"h2d_ms\":" << std::fixed << std::setprecision(4) << h2d << ","
           << "\"exec_ms\":" << exec << ","
           << "\"d2h_ms\":" << d2h << ","
           << "\"profile\":\"" << profile << "\","
           << "\"vram_free_mb\":" << std::fixed << std::setprecision(2) << vram_free << "}";
        
        buffer_.push_back(ss.str());

        if (buffer_.size() >= batch_threshold_) {
            flush();
        }
    }

    void flush() {
        if (buffer_.empty()) return;
        std::ofstream log_file(log_path_, std::ios::app);
        if (log_file.is_open()) {
            for (const auto& line : buffer_) {
                log_file << line << "\n";
            }
            log_file.close();
        }
        buffer_.clear();
    }

    ~TelemetryLogger() {
        flush();
    }
};


// --- WATCHDOG AND TIMEOUT TRACKING REGISTERS ---
std::atomic<uint64_t> g_watchdog_opc_tick{0};      // Mirror counter to verify OPC life
std::atomic<bool>     g_pipeline_state_active{false}; // True when GPU is actively computing a frame
std::atomic<uint64_t> g_step_start_time_ms{0};     // Epoch timestamp of current in-flight frame
std::atomic<int>      g_industrial_fault_code{0};  // 0=OK, 101=OPC_DEAD, 102=GPU_STALL, 103=TRIGGER_STUCK

// --- TIMEOUT CONSTANTS (PRODUCTION MARGINS) ---
constexpr uint64_t OPC_TIMEOUT_MS = 1000;          // Max time OPC thread can go without ticking
constexpr uint64_t GPU_STALL_TIMEOUT_MS = 300;     // Max time a single frame processing window can take
constexpr uint64_t TRIGGER_STUCK_TIMEOUT_MS = 2000; // Max time a PLC trigger can stay high


// Intercepts client write actions to the trigger node inside UaExpert
static void
onInspectionTriggerWrite(UA_Server *server, const UA_NodeId *sessionId, void *sessionContext,
                         const UA_NodeId *nodeId, void *nodeContext, const UA_NumericRange *range,
                         const UA_DataValue *value) {
    if(value->hasValue && UA_Variant_isScalar(&value->value) &&
       value->value.type == &UA_TYPES[UA_TYPES_BOOLEAN]) {
        bool val = *(UA_Boolean*)value->value.data;
        g_opc_trigger.store(val);
    }
}

// Intercepts client write actions to the industrial profile node inside UaExpert/PLC
static void
onIndustrialProfileWrite(UA_Server *server, const UA_NodeId *sessionId, void *sessionContext,
                         const UA_NodeId *nodeId, void *nodeContext, const UA_NumericRange *range,
                         const UA_DataValue *value) {
    if(value->hasValue && UA_Variant_isScalar(&value->value) &&
       value->value.type == &UA_TYPES[UA_TYPES_INT32]) {
        
        int requested_profile = *(UA_Int32*)value->value.data;
        
        if (requested_profile == 10 || requested_profile == 20 || requested_profile == 30) {
            // CRITICAL GUARD: Only log to console if the value actually CHANGED
            if (g_opc_profile_selection.load() != requested_profile) {
                g_opc_profile_selection.store(requested_profile);
                std::cout << "🎛️ [OPC-NETWORK] Dynamic profile CHANGED to: " << requested_profile << std::endl;
            }
            // If it's the same value, we update silently without printing anything
        } else {
            std::cerr << "⚠️ [OPC-NETWORK] Rejected invalid profile code: " << requested_profile << std::endl;
        }
    }
}

// Thread 12: Independent Industrial Server Worker and Telemetry Writer
void opc_server_worker(std::string telemetry_log_path) {
    // Fetch port from environment variable with a standard fallback to 4840
    int opc_port = get_env_int("OPC_UA_PORT", 4840);
    
    std::cout << "🌐 [OPC-NETWORK] Initializing native host server architecture on port " << opc_port << "..." << std::endl;
    
    UA_Server *server = UA_Server_new();
    UA_ServerConfig *config = UA_Server_getConfig(server);

    // Bind the server dynamically to the specified environment port
    UA_ServerConfig_setMinimal(config, opc_port, nullptr);

    // ========================================================================
    // ADDRESS SPACE INITIALIZATION: CREATE THE "PIPELINE" PARENT OBJECT
    // ========================================================================
    UA_ObjectAttributes oAttr = UA_ObjectAttributes_default;
    oAttr.displayName = UA_LOCALIZEDTEXT((char*)"en-US", (char*)"Pipeline");
    UA_NodeId pipelineFolderNodeId = UA_NODEID_STRING_ALLOC(1, (char*)"Pipeline");
    
    UA_Server_addObjectNode(server, 
                            pipelineFolderNodeId,
                            UA_NODEID_NUMERIC(0, UA_NS0ID_OBJECTSFOLDER),
                            UA_NODEID_NUMERIC(0, UA_NS0ID_ORGANIZES),
                            UA_QUALIFIEDNAME_ALLOC(1, (char*)"Pipeline"),
                            UA_NODEID_NUMERIC(0, UA_NS0ID_BASEOBJECTTYPE),
                            oAttr, nullptr, nullptr);

    // ========================================================================
    // REGISTER TELEMETRY VARIABLES
    // ========================================================================

    // Node: Heartbeat
    UA_VariableAttributes hbAttr = UA_VariableAttributes_default;
    int hbVal = 0;
    UA_Variant_setScalar(&hbAttr.value, &hbVal, &UA_TYPES[UA_TYPES_INT32]);
    hbAttr.displayName = UA_LOCALIZEDTEXT((char*)"en-US", (char*)"Heartbeat");
    UA_NodeId hbNodeId = UA_NODEID_STRING_ALLOC(1, (char*)"Pipeline.Heartbeat");
    UA_Server_addVariableNode(server, hbNodeId, pipelineFolderNodeId,
                              UA_NODEID_NUMERIC(0, UA_NS0ID_HASCOMPONENT),
                              UA_QUALIFIEDNAME_ALLOC(1, (char*)"Heartbeat"),
                              UA_NODEID_NUMERIC(0, UA_NS0ID_BASEDATAVARIABLETYPE), hbAttr, nullptr, nullptr);

    // Node: InspectionTrigger
    UA_VariableAttributes trigAttr = UA_VariableAttributes_default;
    bool trigVal = false;
    UA_Variant_setScalar(&trigAttr.value, &trigVal, &UA_TYPES[UA_TYPES_BOOLEAN]);
    trigAttr.displayName = UA_LOCALIZEDTEXT((char*)"en-US", (char*)"InspectionTrigger");
    trigAttr.accessLevel = UA_ACCESSLEVELMASK_READ | UA_ACCESSLEVELMASK_WRITE;
    UA_NodeId trigNodeId = UA_NODEID_STRING_ALLOC(1, (char*)"Pipeline.InspectionTrigger");
    UA_Server_addVariableNode(server, trigNodeId, pipelineFolderNodeId,
                              UA_NODEID_NUMERIC(0, UA_NS0ID_HASCOMPONENT),
                              UA_QUALIFIEDNAME_ALLOC(1, (char*)"InspectionTrigger"),
                              UA_NODEID_NUMERIC(0, UA_NS0ID_BASEDATAVARIABLETYPE), trigAttr, nullptr, nullptr);
    
    UA_ValueCallback callback{nullptr, onInspectionTriggerWrite};
    UA_Server_setVariableNode_valueCallback(server, trigNodeId, callback);

    // Node: InspectionResult
    UA_VariableAttributes resAttr = UA_VariableAttributes_default;
    int resVal = 0;
    UA_Variant_setScalar(&resAttr.value, &resVal, &UA_TYPES[UA_TYPES_INT32]);
    resAttr.displayName = UA_LOCALIZEDTEXT((char*)"en-US", (char*)"InspectionResult");
    UA_NodeId resNodeId = UA_NODEID_STRING_ALLOC(1, (char*)"Pipeline.InspectionResult");
    UA_Server_addVariableNode(server, resNodeId, pipelineFolderNodeId,
                              UA_NODEID_NUMERIC(0, UA_NS0ID_HASCOMPONENT), 
                              UA_QUALIFIEDNAME_ALLOC(1, (char*)"InspectionResult"),
                              UA_NODEID_NUMERIC(0, UA_NS0ID_BASEDATAVARIABLETYPE), resAttr, nullptr, nullptr);

    // Node: AnomalyScore
    UA_VariableAttributes scoreAttr = UA_VariableAttributes_default;
    float scoreVal = 0.0f;
    UA_Variant_setScalar(&scoreAttr.value, &scoreVal, &UA_TYPES[UA_TYPES_FLOAT]);
    scoreAttr.displayName = UA_LOCALIZEDTEXT((char*)"en-US", (char*)"AnomalyScore");
    UA_NodeId scoreNodeId = UA_NODEID_STRING_ALLOC(1, (char*)"Pipeline.AnomalyScore");
    UA_Server_addVariableNode(server, scoreNodeId, pipelineFolderNodeId,
                              UA_NODEID_NUMERIC(0, UA_NS0ID_HASCOMPONENT), 
                              UA_QUALIFIEDNAME_ALLOC(1, (char*)"AnomalyScore"),
                              UA_NODEID_NUMERIC(0, UA_NS0ID_BASEDATAVARIABLETYPE), scoreAttr, nullptr, nullptr);

    // Node: SystemFaultCode
    UA_VariableAttributes faultAttr = UA_VariableAttributes_default;
    int faultVal = 0;
    UA_Variant_setScalar(&faultAttr.value, &faultVal, &UA_TYPES[UA_TYPES_INT32]);
    faultAttr.displayName = UA_LOCALIZEDTEXT((char*)"en-US", (char*)"SystemFaultCode");
    UA_NodeId faultNodeId = UA_NODEID_STRING_ALLOC(1, (char*)"Pipeline.SystemFaultCode");
    UA_Server_addVariableNode(server, faultNodeId, pipelineFolderNodeId,
                              UA_NODEID_NUMERIC(0, UA_NS0ID_HASCOMPONENT), 
                              UA_QUALIFIEDNAME_ALLOC(1, (char*)"SystemFaultCode"),
                              UA_NODEID_NUMERIC(0, UA_NS0ID_BASEDATAVARIABLETYPE), faultAttr, nullptr, nullptr);
    
    //industrial_profile
    UA_VariableAttributes profAttr = UA_VariableAttributes_default;
    int initialProf = g_opc_profile_selection.load();
    UA_Variant_setScalar(&profAttr.value, &initialProf, &UA_TYPES[UA_TYPES_INT32]);
    profAttr.displayName = UA_LOCALIZEDTEXT((char*)"en-US", (char*)"IndustrialProfile");
    
    // Grant both Read and Write capabilities to external clients
    profAttr.accessLevel = UA_ACCESSLEVELMASK_READ | UA_ACCESSLEVELMASK_WRITE;

    UA_NodeId profNodeId = UA_NODEID_STRING_ALLOC(1, (char*)"Pipeline.IndustrialProfile");
    UA_Server_addVariableNode(server, profNodeId, pipelineFolderNodeId,
                              UA_NODEID_NUMERIC(0, UA_NS0ID_HASCOMPONENT),
                              UA_QUALIFIEDNAME_ALLOC(1, (char*)"IndustrialProfile"),
                              UA_NODEID_NUMERIC(0, UA_NS0ID_BASEDATAVARIABLETYPE), profAttr, nullptr, nullptr);

    // Bind our validation write callback to the profile node
    UA_ValueCallback profileCallback{nullptr, onIndustrialProfileWrite};
    UA_Server_setVariableNode_valueCallback(server, profNodeId, profileCallback);
    

    std::ofstream log_stream(telemetry_log_path, std::ios::app);
    auto last_telemetry_dump = std::chrono::steady_clock::now();

    // ========================================================================
    // CRITICAL FIX: PHYSICAL SOCKET LIFECYCLE INITIALIZATION
    // ========================================================================
    UA_StatusCode retval = UA_Server_run_startup(server);
    if(retval != UA_STATUSCODE_GOOD) {
        std::cerr << "❌ Native Host Error: Failed to bind to socket port 4840!" << std::endl;
        UA_Server_delete(server);
        return;
    }

    std::cout << "🌐 Native OPC UA Stack Online. Processing Host Events..." << std::endl;

    // --- MAIN CORE NETWORK ITERATION LOOP ---
    while(g_server_running.load()) {
        UA_Int32 current_result    = static_cast<UA_Int32>(g_opc_result.load());
        UA_Float current_score     = static_cast<UA_Float>(g_opc_anomaly_score.load());
        UA_Boolean current_trigger = static_cast<UA_Boolean>(g_opc_trigger.load());
        UA_Int32 current_fault     = static_cast<UA_Int32>(g_industrial_fault_code.load());
        UA_Int32 current_hb        = static_cast<UA_Int32>(g_opc_heartbeat.load());
        UA_Int32 current_profile   = static_cast<UA_Int32>(g_opc_profile_selection.load()); // NEW
   
        UA_Variant varRes, varScore, varTrig, varFault, varHb, varProf;
        
        UA_Variant_setScalar(&varRes, &current_result, &UA_TYPES[UA_TYPES_INT32]);
        UA_Variant_setScalar(&varScore, &current_score, &UA_TYPES[UA_TYPES_FLOAT]);
        UA_Variant_setScalar(&varTrig, &current_trigger, &UA_TYPES[UA_TYPES_BOOLEAN]);
        UA_Variant_setScalar(&varFault, &current_fault, &UA_TYPES[UA_TYPES_INT32]);
        UA_Variant_setScalar(&varHb, &current_hb, &UA_TYPES[UA_TYPES_INT32]);
        UA_Variant_setScalar(&varProf, &current_profile, &UA_TYPES[UA_TYPES_INT32]); // NEW

        UA_Server_writeValue(server, resNodeId, varRes);
        UA_Server_writeValue(server, scoreNodeId, varScore);
        UA_Server_writeValue(server, trigNodeId, varTrig);
        UA_Server_writeValue(server, faultNodeId, varFault);
        UA_Server_writeValue(server, hbNodeId, varHb);
        UA_Server_writeValue(server, profNodeId, varProf); // NEW
        
       // Drives network lifecycle loops natively on host loopback card
        UA_Server_run_iterate(server, false);
        g_watchdog_opc_tick.fetch_add(1);

        // --- NEW STRATEGIC FIX: YIELD SERVER THREAD CORE ---
        // Industrial servers typically iterate network stacks every 2ms to 5ms.
        // This stops thread starvation and satisfies your watchdog easily.
        std::this_thread::sleep_for(std::chrono::milliseconds(2)); 
        
        // --- UPGRADED DECOUPLED SYSTEM HEALTH HEARTBEAT LOGGER ---
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_telemetry_dump).count() >= 5) {
            // 1. Collect VRAM Statistics via CUDA
            size_t free_mem = 0, total_mem = 0;
            cudaMemGetInfo(&free_mem, &total_mem);
            double vram_free_mb = static_cast<double>(free_mem) / (1024.0 * 1024.0);

            // 2. Collect Host RAM Statistics via OS filesystem
            double host_ram_mb = get_host_ram_usage_mb();

            // 3. Write structured JSONL record representing immediate systemic state
            if (log_stream.is_open()) {
                log_stream << "{\"timestamp\":\"" << get_iso_timestamp() << "\","
                           << "\"record_type\":\"SYSTEM_HEARTBEAT\","
                           << "\"opc_ticks\":" << g_watchdog_opc_tick.load() << ","
                           << "\"heartbeat\":" << current_hb << ","
                           << "\"fault_code\":" << current_fault << ","
                           << "\"host_ram_usage_mb\":" << std::fixed << std::setprecision(2) << host_ram_mb << ","
                           << "\"vram_free_mb\":" << vram_free_mb << ","
                           << "\"active_profile_code\":" << current_profile << "}\n";
                log_stream.flush(); // Force immediate disk flush to maintain logging safety
            }
            last_telemetry_dump = now;
        }
    }

    // ========================================================================
    // CRITICAL FIX: SHUTDOWN FLUSH
    // ========================================================================
    UA_Server_run_shutdown(server); 

    if(log_stream.is_open()) log_stream.close();
    UA_Server_delete(server);
    std::cout << "🌐 OPC UA Server natively terminated and port released." << std::endl;
}

// Helper function to fetch the current epoch time in milliseconds
uint64_t get_current_time_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
}

// Thread 13: Independent Watchdog Sentinel
void industrial_watchdog_worker() {
    uint64_t last_opc_tick = g_watchdog_opc_tick.load();
    auto last_check_time = std::chrono::steady_clock::now();

    std::cout << "🛡️ Industrial Safety Watchdog (Thread 13) Active." << std::endl;

    while (g_server_running.load()) {
        // Run the watchdog checks every 100ms to keep CPU usage minimal
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        uint64_t current_time = get_current_time_ms();

        // --- CHECK 1: OPC UA THREAD ALIVE VERIFICATION ---
        uint64_t current_opc_tick = g_watchdog_opc_tick.load();
        if (current_opc_tick == last_opc_tick) {
            auto now = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_check_time).count();
            
            if (duration >= OPC_TIMEOUT_MS && g_industrial_fault_code.load() == 0) {
                std::cerr << "🚨 [WATCHDOG CRITICAL] OPC UA THREAD STALLED OR DEAD!" << std::endl;
                g_industrial_fault_code.store(101); // 101 = OPC_LAYER_CRASH
            }
        } else {
            // The OPC thread updated its counter, so reset our tracking registers
            last_opc_tick = current_opc_tick;
            last_check_time = std::chrono::steady_clock::now();
        }

        // --- CHECK 2: GPU INFERENCE / STREAM STALL MONITORING ---
        if (g_pipeline_state_active.load()) {
            uint64_t step_start = g_step_start_time_ms.load();
            if (current_time > step_start) {
                uint64_t process_duration = current_time - step_start;
                
                if (process_duration > GPU_STALL_TIMEOUT_MS && g_industrial_fault_code.load() == 0) {
                    std::cerr << "🚨 [WATCHDOG CRITICAL] CUDA PIPELINE OR STREAM STALL DETECTED! Duration: " 
                              << process_duration << "ms" << std::endl;
                    g_industrial_fault_code.store(102); // 102 = GPU_COMPUTE_STALL
                }
            }
        }

        // --- CHECK 3: PLC TRIGGERS STUCK HIGH ---
        if (g_opc_trigger.load()) {
            uint64_t trigger_start = g_step_start_time_ms.load();
            if (current_time > trigger_start && g_pipeline_state_active.load()) {
                uint64_t trigger_duration = current_time - trigger_start;
                
                if (trigger_duration > TRIGGER_STUCK_TIMEOUT_MS && g_industrial_fault_code.load() == 0) {
                    std::cerr << "🚨 [WATCHDOG CRITICAL] PLC HANDSHAKE TRIGGER STUCK HIGH!" << std::endl;
                    g_industrial_fault_code.store(103); // 103 = TRIGGER_STUCK_FAULT
                    g_opc_trigger.store(false);         // Force-clear latch to save tracking mechanics
                }
            }
        }
    }
    std::cout << "🛡️ Watchdog sentinel safely spun down." << std::endl;
}

int main() { 
    // Ingest all configurations directly from Docker runtime variables with fallback defaults
    std::string engine_path        = get_env_var("MODEL_ENGINE_PATH", "/app/models/pcb_fastflow_fp16.engine");
    std::string telemetry_log_path = get_env_var("TELEMETRY_LOG_PATH", "/app/logs/telemetry_production.json");
    // Instantiate the production telemetry batching mechanism
    TelemetryLogger telemetry(telemetry_log_path);
    std::cout << "📝 Production telemetry engine online logging to: " << telemetry_log_path << std::endl;

    // Dynamic Deployment Profile selection (10 = SAFETY, 20 = BALANCED, 30 = UPTIME)
    int profile_env = get_env_int("DEPLOYMENT_PROFILE", 20);
    g_opc_profile_selection.store(profile_env);

    IndustrialProfile current_profile = IndustrialProfile::BALANCED;
    float threshold = -0.2500f; 
    std::string profile_str = "BALANCED";

    if (profile_env == 10) {
        current_profile = IndustrialProfile::SAFETY;
        threshold = -0.3200f;
        profile_str = "SAFETY";
    } else if (profile_env == 30) {
        current_profile = IndustrialProfile::UPTIME;
        threshold = -0.1500f;
        profile_str = "UPTIME";
    }

    std::cout << "🚀 Engine booting under [" << profile_str << "] profile. Target Threshold: " << threshold << std::endl;


    std::ifstream file(engine_path, std::ios::binary); 
    if (!file.good()) { std::cerr << "Engine file verification failed!" << std::endl; return -1; } 
     
    file.seekg(0, file.end); 
    size_t size = file.tellg(); 
    file.seekg(0, file.beg); 
    std::vector<char> engine_data(size); 
    file.read(engine_data.data(), size); 

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger); 
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engine_data.data(), size); 
    nvinfer1::IExecutionContext* context = engine->createExecutionContext(); 

    // DOUBLE BUFFER STRUCTURE INITIALIZATION
    void* buffers[2][2]; 
    cudaMalloc(&buffers[0][0], INPUT_SIZE);  
    cudaMalloc(&buffers[0][1], OUTPUT_SIZE); 
    cudaMalloc(&buffers[1][0], INPUT_SIZE);  
    cudaMalloc(&buffers[1][1], OUTPUT_SIZE); 

    float* h_output_pinned[2] = {nullptr, nullptr}; 
    cudaHostAlloc(&h_output_pinned[0], OUTPUT_SIZE, cudaHostAllocPortable); 
    cudaHostAlloc(&h_output_pinned[1], OUTPUT_SIZE, cudaHostAllocPortable); 
    
    cv::Mat h_anomaly[2];
    h_anomaly[0] = cv::Mat(INPUT_H, INPUT_W, CV_32FC1, h_output_pinned[0]);
    h_anomaly[1] = cv::Mat(INPUT_H, INPUT_W, CV_32FC1, h_output_pinned[1]);

    cv::cuda::GpuMat d_frame[2];  
    cv::cuda::GpuMat d_frame_rgba[2];
    cv::cuda::GpuMat d_warped_320[2]  = { cv::cuda::GpuMat(320, 320, CV_8UC4), cv::cuda::GpuMat(320, 320, CV_8UC4) };   
    cv::cuda::GpuMat d_rotated_320[2] = { cv::cuda::GpuMat(320, 320, CV_8UC4), cv::cuda::GpuMat(320, 320, CV_8UC4) };  
    cv::cuda::GpuMat d_final_padded[2];
    cv::cuda::GpuMat d_final_256[2]   = { cv::cuda::GpuMat(INPUT_H, INPUT_W, CV_8UC3), cv::cuda::GpuMat(INPUT_H, INPUT_W, CV_8UC3) }; 

    std::vector<cv::cuda::GpuMat> bgra_channels[2]; 
    std::vector<cv::cuda::GpuMat> bgr_channels[2] = { std::vector<cv::cuda::GpuMat>(3), std::vector<cv::cuda::GpuMat>(3) };

    cv::cuda::GpuMat d_anomaly_map_blurred[2] = { cv::cuda::GpuMat(INPUT_H, INPUT_W, CV_32FC1), cv::cuda::GpuMat(INPUT_H, INPUT_W, CV_32FC1) }; 
    cv::cuda::GpuMat d_anomaly_map_final[2]   = { cv::cuda::GpuMat(INPUT_H, INPUT_W, CV_32FC1), cv::cuda::GpuMat(INPUT_H, INPUT_W, CV_32FC1) }; 

    cv::cuda::GpuMat d_hsv[2], d_mask[2], d_proj_cols[2];
    cv::Mat h_proj_cols[2];

    cv::cuda::Stream stream[2]; 
    cudaStream_t raw_stream[2];
    raw_stream[0] = cv::cuda::StreamAccessor::getStream(stream[0]);
    raw_stream[1] = cv::cuda::StreamAccessor::getStream(stream[1]);

    cv::Ptr<cv::cuda::Filter> gaussian_filter = cv::cuda::createGaussianFilter(CV_32FC1, CV_32FC1, cv::Size(5, 5), 0); 

    // --- TRACKING PARAMETERS FOR IN-FLIGHT TIMING INTERLOCKS ---
    struct InFlightMeta {
        std::string filename;
        int label;
        double io_read_ms;
        double preprocess_ms;
        double h2d_ms;
        double inference_ms;
        double postprocess_ms;
    } inflight_meta[2];

    std::cout << "🔥 Calibrating CUDA Streams & Warming up GPU (20 Tensors)..." << std::endl;
    cudaMemsetAsync(buffers[0][0], 0, INPUT_SIZE, raw_stream[0]);
    cudaMemsetAsync(buffers[1][0], 0, INPUT_SIZE, raw_stream[1]);
    for (int w = 0; w < 10; ++w) {
        context->enqueueV2(buffers[0], raw_stream[0], nullptr);
        context->enqueueV2(buffers[1], raw_stream[1], nullptr);
    }
    cudaStreamSynchronize(raw_stream[0]);
    cudaStreamSynchronize(raw_stream[1]);
    std::cout << "✅ Warmup Complete. GPU Stable. Handshake Code: " << static_cast<int>(current_profile) << std::endl;
    
    // --- ZONE 4.1: BOOT INDUSTRIAL CONTROL AND SAFETY THREADS ---
    std::thread opc_thread(opc_server_worker, telemetry_log_path);
    std::thread watchdog_thread(industrial_watchdog_worker);
    std::cout << "🌐 Industrial network services active. opc.tcp:/ /0.0.0.0:4840" << std::endl;


    // Ingest the target acquisition folder or device path where live images are dropped
    std::string live_capture_path = get_env_var("LIVE_CAPTURE_PATH", "/app/capture/live_pcb.png");

    std::cout << "📥 Monitoring target production capture frame path: " << live_capture_path << std::endl;
    auto pipeline_start_time = std::chrono::high_resolution_clock::now();

    int idx = 0; 
    bool first_iterations[2] = {true, true}; 

    // --- MAIN ENGINE PIPELINE EXECUTION LOOP --- 
    while (g_server_running.load()) { 
        // 1. Line blocks here until an external PLC toggles the inspection trigger via OPC UA
        while (!g_opc_trigger.load() && g_server_running.load()) {
            std::this_thread::sleep_for(std::chrono::microseconds(200)); 
        }

        if (!g_server_running.load()) break;

        // 2. Ingest the newly available physical frame data dropped by the camera hardware
        auto t_io_start = std::chrono::high_resolution_clock::now();
        cv::Mat live_frame = cv::imread(live_capture_path);
        auto t_io_end = std::chrono::high_resolution_clock::now();

        if (live_frame.empty()) {
            std::cerr << "⚠️ [INPUT ERROR] Failed to read live frame at: " << live_capture_path << std::endl;
            g_industrial_fault_code.store(104); // 104 = CAMERA_CAPTURE_FAULT
            g_opc_trigger.store(false);         // Prevent lockups by releasing latch
            continue;
        }

        // Map live frame parameters to a structured operational payload
        FrameData job;
        job.frame = live_frame;
        job.filename = "LIVE_INSPECTION_FRAME";
        job.label = 0; // Production default (Unlabeled)
        job.io_read_ms = std::chrono::duration<double, std::milli>(t_io_end - t_io_start).count();

        // Arm the safety watchdog stopwatch for this frame
        g_step_start_time_ms.store(get_current_time_ms());
        g_pipeline_state_active.store(true);
        auto t_pre_start = std::chrono::high_resolution_clock::now();

        cv::Mat hsv, mask; 
        cv::cvtColor(job.frame, hsv, cv::COLOR_BGR2HSV); 
        cv::inRange(hsv, cv::Scalar(90, 50, 50), cv::Scalar(135, 255, 255), mask); 
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7)); 
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel); 

        std::vector<std::vector<cv::Point>> contours; 
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE); 
        if (contours.empty()) {
            std::cout << "DEBUG: [SKIP] No contours found for: " << job.filename << std::endl;
            continue; 
        }

        auto largest = *std::max_element(contours.begin(), contours.end(),  
            [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {  
                return cv::contourArea(a) < cv::contourArea(b);  
            }); 

        cv::RotatedRect rect = cv::minAreaRect(largest); 
        
        // HIGH-SPEED CPU EARLY REJECT GATEWAY (METRIC CALIBRATION INTERCEPT)
        // ========================================================================
        float w = rect.size.width;
        float h = rect.size.height;
        
        // Handle potential zero division safety checks gracefully
        if (w == 0 || h == 0) {
            std::cout << "⚠️ [" << job.filename << "] INVALID GEOMETRY ENCOUNTERED -> REJECTING" << std::endl;
            continue;
        }

        double calc_ar = std::max(w, h) / std::min(w, h);
        double calc_scale = w * h;

        // Apply your tight statistical calibration threshold cuts
        if (calc_ar < 2.00 || calc_ar > 2.50 || calc_scale < 300000.0 || calc_scale > 450000.0) {
            // Log to telemetry using the established struct rules
            size_t free_mem = 0, total_mem = 0;
            cudaMemGetInfo(&free_mem, &total_mem);
            double vram_free_mb = static_cast<double>(free_mem) / (1024.0 * 1024.0);

            telemetry.log_event(
                job.filename,          // Board identity context
                "MECHANICAL_ERROR",    // Structural state trigger signature
                profile_str,           // Operational profile mapping
                0.0, 0.0, 0.0,         // Zero out GPU metrics since execution bypassed it
                vram_free_mb
            );

            // Establish directory path and dump the raw frame safely to storage for manual QA auditing
            std::string err_dir = get_env_var("ERROR_DUMP_DIR", "/app/logs/mechanical_errors/");
            fs::create_directories(err_dir);
            cv::imwrite(err_dir + "ERR_ME01_" + get_iso_timestamp() + ".png", job.frame);

            // --- ZONE 4.3: EARLY REJECT MECHANICAL FAULT INTERLOCK ---
            g_opc_result.store(2);               // 2 = MECHANICAL_ERROR status code
            g_pipeline_state_active.store(false); // Disarm watchdog stopwatch
            g_opc_trigger.store(false);          // Release handshake latch, clearing trigger for PLC
            
            continue;
        }
        
        cv::Point2f pts[4]; 
        rect.points(pts); 

        std::vector<cv::Point2f> src_pts(4); 
        std::vector<cv::Point2f> box(pts, pts + 4); 
        auto sum_c = [](cv::Point2f a, cv::Point2f b) { return (a.x + a.y) < (b.x + b.y); }; 
        auto diff_c = [](cv::Point2f a, cv::Point2f b) { return (a.y - a.x) < (b.y - b.x); }; 
         
        src_pts[0] = *std::min_element(box.begin(), box.end(), sum_c);  
        src_pts[1] = *std::min_element(box.begin(), box.end(), diff_c); 
        src_pts[2] = *std::max_element(box.begin(), box.end(), sum_c); 
        src_pts[3] = *std::max_element(box.begin(), box.end(), diff_c); 

        float d01 = cv::norm(src_pts[0] - src_pts[1]); 
        float d03 = cv::norm(src_pts[0] - src_pts[3]); 
        float board_w, board_h; 

        if (d01 == 0 || d03 == 0) {
            std::cout << "DEBUG: [SKIP] Invalid norm dimension for: " << job.filename << std::endl;
            continue;
        }

        if (d03 > d01) { 
            std::rotate(src_pts.begin(), src_pts.begin() + 1, src_pts.end()); 
            board_w = d03; board_h = d01; 
        } else { 
            board_w = d01; board_h = d03; 
        } 

        cv::Mat mom_mask = ~mask;  
        cv::Moments mom = cv::moments(mom_mask, true); 
        bool do_rotate = (mom.m00 > 0 && (mom.m01 / mom.m00) < (job.frame.rows / 2.0f)); 

        float target_w = 240.0f;  
        float aspect = board_h / board_w;  
        float target_h = target_w * aspect;  
        float ox = (320.0f - target_w) / 2.0f; 
        float oy = (320.0f - target_h) / 2.0f; 

        std::vector<cv::Point2f> dst_pts(4); 
        dst_pts[0] = {ox,            oy}; 
        dst_pts[1] = {ox + target_w, oy}; 
        dst_pts[2] = {ox + target_w, oy + target_h}; 
        dst_pts[3] = {ox,            oy + target_h}; 

        cv::Mat M = cv::getPerspectiveTransform(src_pts, dst_pts); 

        // ========================================================================
        // PING-PONG SLOT SYNC POINT & ACCUMULATION PROFILE REGISTRATION
        // ========================================================================
        if (!first_iterations[idx]) {
            auto t_d2h_start = std::chrono::high_resolution_clock::now();
            stream[idx].waitForCompletion();
            auto t_d2h_end = std::chrono::high_resolution_clock::now();
            double d2h_ms = std::chrono::duration<double, std::milli>(t_d2h_end - t_d2h_start).count();

            double max_val; 
            cv::minMaxLoc(h_anomaly[idx], nullptr, &max_val); 
            
            size_t free_mem = 0, total_mem = 0;
            cudaMemGetInfo(&free_mem, &total_mem);
            double vram_free_mb = static_cast<double>(free_mem) / (1024.0 * 1024.0);
            
            // --- DYNAMIC PER-FRAME PROFILE AND THRESHOLD EVALUATION ---
            int active_profile_code = g_opc_profile_selection.load();
            float frame_threshold = -0.2500f; // Default Balanced
            std::string active_profile_str = "BALANCED";

            if (active_profile_code == 10) {
                frame_threshold = -0.3200f;   // Safety Mode (Tighter bounds)
                active_profile_str = "SAFETY";
            } else if (active_profile_code == 30) {
                frame_threshold = -0.1500f;   // Uptime Mode (More lenient)
                active_profile_str = "UPTIME";
            }

            // Run validation check using live parameters
            bool predicted_anomaly = (max_val > frame_threshold);
            std::string status_str = predicted_anomaly ? "FAIL" : "PASS";

            // --- ZONE 4.4: MODEL PREDICTION HANDSHAKE RELEASES (MAIN LOOP) ---
            int dynamic_status_code = predicted_anomaly ? 1 : 0; // 1 = FAIL, 0 = PASS
            g_opc_result.store(dynamic_status_code);
            g_opc_anomaly_score.store(static_cast<float>(max_val));
            
            g_pipeline_state_active.store(false); // Disarm watchdog stopwatch
            g_opc_heartbeat.fetch_add(1);         // Increment verification ticker
            g_opc_trigger.store(false);          // Release handshake latch, signaling completion to PLC

            
            // Write detailed inference execution times to our asynchronous logger batch
            telemetry.log_event(
                inflight_meta[idx].filename,
                status_str,
                active_profile_str,
                inflight_meta[idx].h2d_ms,
                inflight_meta[idx].inference_ms,
                d2h_ms,
                vram_free_mb
            );
            
#if ENGINE_BENCHMARK_MODE
            {
                std::lock_guard<std::mutex> lock(g_metrics_mutex);
                g_preprocess_lats.push_back(inflight_meta[idx].preprocess_ms);
                g_h2d_lats.push_back(inflight_meta[idx].h2d_ms);
                g_inference_lats.push_back(inflight_meta[idx].inference_ms);
                g_postprocess_lats.push_back(inflight_meta[idx].postprocess_ms);
                g_d2h_lats.push_back(d2h_ms);
            }
#endif

        }

        // --- ENQUEUE ASYNC STAGES ON STREAM[IDX] --- 
        d_frame[idx].upload(job.frame, stream[idx]); 
        cv::cuda::cvtColor(d_frame[idx], d_frame_rgba[idx], cv::COLOR_BGR2BGRA, 0, stream[idx]); 

        cv::cuda::warpPerspective(d_frame_rgba[idx], d_warped_320[idx], M, cv::Size(320, 320), 
                                  cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0,0,0,0), stream[idx]); 

        if (do_rotate) { 
            cv::cuda::flip(d_warped_320[idx], d_warped_320[idx], -1, stream[idx]); 
        } 

        cv::cuda::transpose(d_warped_320[idx], d_rotated_320[idx], stream[idx]); 
        cv::cuda::flip(d_rotated_320[idx], d_rotated_320[idx], 0, stream[idx]);  

        cv::Rect roi_box(32, 32, INPUT_W, INPUT_H); 
        d_final_padded[idx] = d_rotated_320[idx](roi_box);  

        cv::cuda::split(d_final_padded[idx], bgra_channels[idx], stream[idx]);  
        bgr_channels[idx][0] = bgra_channels[idx][0]; 
        bgr_channels[idx][1] = bgra_channels[idx][1]; 
        bgr_channels[idx][2] = bgra_channels[idx][2]; 
        cv::cuda::merge(bgr_channels[idx], d_final_256[idx], stream[idx]); 

        cv::cuda::cvtColor(d_final_256[idx], d_hsv[idx], cv::COLOR_BGR2HSV, 0, stream[idx]); 
        cv::cuda::inRange(d_hsv[idx], cv::Scalar(90, 50, 50), cv::Scalar(135, 255, 255), d_mask[idx], stream[idx]); 
        cv::cuda::reduce(d_mask[idx], d_proj_cols[idx], 0, cv::REDUCE_SUM, CV_32S, stream[idx]); 
         
        d_proj_cols[idx].download(h_proj_cols[idx], stream[idx]); 
        stream[idx].waitForCompletion(); // Local 1D Edge dependency sync barrier

        int detected_l_wall = -1; 
        int detected_r_wall = -1; 
        const int* proj_ptr = h_proj_cols[idx].ptr<int>(); 

        for (int c = 0; c < INPUT_W; ++c) { 
            if (proj_ptr[c] > 0) { 
                if (detected_l_wall == -1) detected_l_wall = c; 
                detected_r_wall = c; 
            } 
        } 

        int current_board_width = detected_r_wall - detected_l_wall; 
        if (detected_l_wall == -1 || current_board_width < VALID_MIN_WIDTH_THRESHOLD) { 
            std::cout << "⚠️ [" << job.filename << "] EARLY REJECT" << std::endl; 
            continue;  
        } 

        int final_left_wall  = detected_l_wall - 5; 
        int final_right_wall = detected_r_wall + 5; 

        auto t_pre_end = std::chrono::high_resolution_clock::now();
        double preprocess_ms = std::chrono::duration<double, std::milli>(t_pre_end - t_pre_start).count();

        // --- ASYNC HIGH-PERFORMANCE INFERENCE OVERLAP ENQUEUING ---
        auto t_h2d_start = std::chrono::high_resolution_clock::now();
        launch_hwc_to_chw_interface(d_final_256[idx].data, static_cast<float*>(buffers[idx][0]),  
                                    d_final_256[idx].cols, d_final_256[idx].rows, d_final_256[idx].step,  
                                    raw_stream[idx]); 
        auto t_h2d_end = std::chrono::high_resolution_clock::now();
        double h2d_ms = std::chrono::duration<double, std::milli>(t_h2d_end - t_h2d_start).count();

        auto t_inf_start = std::chrono::high_resolution_clock::now();
        context->enqueueV2(buffers[idx], raw_stream[idx], nullptr); 
        auto t_inf_end = std::chrono::high_resolution_clock::now();
        double inference_ms = std::chrono::duration<double, std::milli>(t_inf_end - t_inf_start).count();

        auto t_post_start = std::chrono::high_resolution_clock::now();
        gaussian_filter->apply(cv::cuda::GpuMat(INPUT_H, INPUT_W, CV_32FC1, buffers[idx][1]), d_anomaly_map_blurred[idx], stream[idx]); 
        
        launch_postprocess_mask_amp( 
            reinterpret_cast<float*>(d_anomaly_map_blurred[idx].data), 
            reinterpret_cast<float*>(d_anomaly_map_final[idx].data), 
            INPUT_W, INPUT_H, 
            final_left_wall, final_right_wall, 
            Y_START, Y_END, X_START, X_END, 
            OSC_Y_START, OSC_Y_END, OSC_X_START, OSC_X_END, 
            AMP_TRIGGER_THRESHOLD, PIN_ZONE_BIAS, OSC_BIAS, 
            raw_stream[idx] 
        ); 
        auto t_post_end = std::chrono::high_resolution_clock::now();
        double post_ms = std::chrono::duration<double, std::milli>(t_post_end - t_post_start).count();

	cudaMemcpyAsync(h_anomaly[idx].data, d_anomaly_map_final[idx].data, OUTPUT_SIZE, cudaMemcpyDeviceToHost, raw_stream[idx]); 

        inflight_meta[idx] = { job.filename, job.label, job.io_read_ms, preprocess_ms, h2d_ms, inference_ms, post_ms };
        first_iterations[idx] = false;

        idx = !idx; 
    } 

    std::cout << "DEBUG: Main loop broke! Draining pipeline trails..." << std::endl;

    // ========================================================================
    // CONCURRENT PIPELINE FLUSH BARRIER (Final Asymmetric Drain)
    // ========================================================================
    for (int final_slot = 0; final_slot < 2; ++final_slot) {
        if (!first_iterations[final_slot]) {
            auto t_d2h_start = std::chrono::high_resolution_clock::now();
            stream[final_slot].waitForCompletion();
            auto t_d2h_end = std::chrono::high_resolution_clock::now();
            double d2h_ms = std::chrono::duration<double, std::milli>(t_d2h_end - t_d2h_start).count();

            double max_val; 
            cv::minMaxLoc(h_anomaly[final_slot], nullptr, &max_val); 
            
            size_t free_mem = 0, total_mem = 0;
            cudaMemGetInfo(&free_mem, &total_mem);
            double vram_free_mb = static_cast<double>(free_mem) / (1024.0 * 1024.0);

            // --- DYNAMIC PER-FRAME PROFILE AND THRESHOLD EVALUATION ---
            int active_profile_code = g_opc_profile_selection.load();
            float frame_threshold = -0.2500f; // Default Balanced
            std::string active_profile_str = "BALANCED";

            if (active_profile_code == 10) {
                frame_threshold = -0.3200f;   // Safety Mode (Tighter bounds)
                active_profile_str = "SAFETY";
            } else if (active_profile_code == 30) {
                frame_threshold = -0.1500f;   // Uptime Mode (More lenient)
                active_profile_str = "UPTIME";
            }

            // Run validation check using live parameters
            bool predicted_anomaly = (max_val > frame_threshold);
            std::string status_str = predicted_anomaly ? "FAIL" : "PASS";

            // --- ZONE 4.4: MODEL PREDICTION HANDSHAKE RELEASES (MAIN LOOP) ---
            int dynamic_status_code = predicted_anomaly ? 1 : 0; // 1 = FAIL, 0 = PASS
            g_opc_result.store(dynamic_status_code);
            g_opc_anomaly_score.store(static_cast<float>(max_val));
            
            g_pipeline_state_active.store(false); // Disarm watchdog stopwatch
            g_opc_heartbeat.fetch_add(1);         // Increment verification ticker
            g_opc_trigger.store(false);          // Release handshake latch, signaling completion to PLC

            // Write detailed inference execution times to our asynchronous logger batch
            

            telemetry.log_event(
                inflight_meta[idx].filename,
                status_str,
                active_profile_str,
                inflight_meta[idx].h2d_ms,
                inflight_meta[idx].inference_ms,
                d2h_ms,
                vram_free_mb
            );
            

#if ENGINE_BENCHMARK_MODE
            {
                std::lock_guard<std::mutex> lock(g_metrics_mutex);
                g_preprocess_lats.push_back(inflight_meta[final_slot].preprocess_ms);
                g_h2d_lats.push_back(inflight_meta[final_slot].h2d_ms);
                g_inference_lats.push_back(inflight_meta[final_slot].inference_ms);
                g_postprocess_lats.push_back(inflight_meta[final_slot].postprocess_ms);
                g_d2h_lats.push_back(d2h_ms);
            }
#endif

        }
    }


#if ENGINE_BENCHMARK_MODE
    auto pipeline_end_time = std::chrono::high_resolution_clock::now();
    double total_wall_time_s = std::chrono::duration<double>(pipeline_end_time - pipeline_start_time).count();

    // --- QUANTILE ACCUMULATOR MATH COMPILER FOR REPORT ---
    auto compute_stats = [](const std::vector<double>& v, double& avg, double& max_val) {
        if (v.empty()) { avg = 0.0; max_val = 0.0; return; }
        avg = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        max_val = *std::max_element(v.begin(), v.end());
    };

    double io_avg, io_max, pre_avg, pre_max, h2d_avg, h2d_max, inf_avg, inf_max, post_avg, post_max, d2h_avg, d2h_max;
    compute_stats(g_io_read_lats, io_avg, io_max);
    compute_stats(g_preprocess_lats, pre_avg, pre_max);
    compute_stats(g_h2d_lats, h2d_avg, h2d_max);
    compute_stats(g_inference_lats, inf_avg, inf_max);
    compute_stats(g_postprocess_lats, post_avg, post_max);
    compute_stats(g_d2h_lats, d2h_avg, d2h_max);

    std::cout << "\n" << std::string(65, '=') << "\n";
    std::cout << "                STAGEWISE LATENCY EVALUATION REPORT\n";
    std::cout << std::string(65, '-') << "\n";
    std::cout << std::left << std::setw(25) << "STAGE" << " | " << std::setw(20) << "AVG LATENCY (ms)" << " | " << std::setw(10) << "MAX (ms)" << "\n";
    std::cout << std::string(65, '-') << "\n";
    std::cout << std::left << std::setw(25) << "IO_DISK_READ" << " | " << std::setw(20) << io_avg << " | " << std::setw(10) << io_max << "\n";
    std::cout << std::left << std::setw(25) << "CPU_PREPROCESS_WARP" << " | " << std::setw(20) << pre_avg << " | " << std::setw(10) << pre_max << " (Overlap Active)\n";
    std::cout << std::left << std::setw(25) << "GPU_H2D_TRANSFER" << " | " << std::setw(20) << h2d_avg << " | " << std::setw(10) << h2d_max << "\n";
    std::cout << std::left << std::setw(25) << "GPU_ENGINE_EXECUTION" << " | " << std::setw(20) << inf_avg << " | " << std::setw(10) << inf_max << "\n";
    std::cout << std::left << std::setw(25) << "GPU_POSTPROCESS_MASK" << " | " << std::setw(20) << post_avg << " | " << std::setw(10) << post_max << "\n";
    std::cout << std::left << std::setw(25) << "GPU_D2H_TRANSFER" << " | " << std::setw(20) << d2h_avg << " | " << std::setw(10) << d2h_max << "\n";
    std::cout << std::string(65, '=') << "\n\n";
#endif

    // Assets Deallocation
    cudaFree(buffers[0][0]); cudaFree(buffers[0][1]);
    cudaFree(buffers[1][0]); cudaFree(buffers[1][1]);
    cudaFreeHost(h_output_pinned[0]); 
    cudaFreeHost(h_output_pinned[1]); 
    delete context;  
    delete engine;  
    delete runtime;

    // --- ZONE 4.5: GRACAFUL INDUSTRIAL SHUTDOWN CLEANUP ---
    g_server_running.store(false);
    
    if (watchdog_thread.joinable())  watchdog_thread.join();
    if (opc_thread.joinable())       opc_thread.join();
    
    std::cout << "🛡️ System-wide industrial context safely dissolved." << std::endl; 
    return 0; 
}
