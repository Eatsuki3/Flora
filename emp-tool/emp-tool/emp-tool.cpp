#include "emp-tool/emp-tool/execution/circuit_execution.h"
#include "emp-tool/emp-tool/execution/protocol_execution.h"

// #ifndef THREADING
// emp::ProtocolExecution* emp::ProtocolExecution::prot_exec = nullptr;
// emp::CircuitExecution* emp::CircuitExecution::circ_exec = nullptr;
// #else
thread_local emp::ProtocolExecution* emp::ProtocolExecution::prot_exec =
    nullptr;
thread_local emp::CircuitExecution* emp::CircuitExecution::circ_exec = nullptr;
// #endif
