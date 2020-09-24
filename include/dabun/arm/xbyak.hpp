#pragma once

#include "xbyak_aarch64/xbyak_aarch64.h"

using xbyak_buffer_type = std::uint32_t;

namespace Xbyak
{
using namespace Xbyak_aarch64;
using CodeArray     = CodeArrayAArch64;
using Allocator     = AllocatorAArch64;
using CodeGenerator = CodeGeneratorAArch64;
using Reg64         = XReg;
using Label         = LabelAArch64;
} // namespace Xbyak
