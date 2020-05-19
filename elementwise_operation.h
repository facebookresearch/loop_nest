#pragma once

#define XBYAK_NO_OP_NAMES
#include "xbyak/xbyak.h"

#include "isa.h"

namespace facebook
{
namespace sysml
{
namespace aot
{

class elementwise_operation
{
public:
    virtual ~elementwise_operation(){};

    virtual void initialize_vector(Xbyak::CodeGenerator*,
                                   std::vector<Xbyak::Zmm> const&, avx512) = 0;
    virtual void initialize_vector(Xbyak::CodeGenerator*,
                                   std::vector<Xbyak::Ymm> const&,
                                   avx2_plus)                              = 0;
    virtual void initialize_vector(Xbyak::CodeGenerator*,
                                   std::vector<Xbyak::Ymm> const&, avx2)   = 0;

    virtual void initialize_scalar(Xbyak::CodeGenerator*,
                                   std::vector<Xbyak::Xmm> const&, avx512) = 0;
    virtual void initialize_scalar(Xbyak::CodeGenerator*,
                                   std::vector<Xbyak::Xmm> const&,
                                   avx2_plus)                              = 0;
    virtual void initialize_scalar(Xbyak::CodeGenerator*,
                                   std::vector<Xbyak::Xmm> const&, avx2)   = 0;

    virtual void process_vector(Xbyak::CodeGenerator*, Xbyak::Zmm const&,
                                std::vector<Xbyak::Zmm> const&, avx512)    = 0;
    virtual void process_vector(Xbyak::CodeGenerator*, Xbyak::Ymm const&,
                                std::vector<Xbyak::Ymm> const&, avx2_plus) = 0;
    virtual void process_vector(Xbyak::CodeGenerator*, Xbyak::Ymm const&,
                                std::vector<Xbyak::Ymm> const&, avx2)      = 0;

    virtual void process_scalar(Xbyak::CodeGenerator*, Xbyak::Xmm const&,
                                std::vector<Xbyak::Xmm> const&, avx512)    = 0;
    virtual void process_scalar(Xbyak::CodeGenerator*, Xbyak::Xmm const&,
                                std::vector<Xbyak::Xmm> const&, avx2_plus) = 0;
    virtual void process_scalar(Xbyak::CodeGenerator*, Xbyak::Xmm const&,
                                std::vector<Xbyak::Xmm> const&, avx2)      = 0;
};

class relu_elementwise_operation : public elementwise_operation
{
private:
    int auxillary_index = 0;

public:
    void initialize_vector(Xbyak::CodeGenerator*          cg,
                           std::vector<Xbyak::Zmm> const& auxillary,
                           avx512) override
    {
        assert(auxillary.size());
        auxillary_index = auxillary[0].getIdx();
        cg->vxorpd(auxillary[0], auxillary[0], auxillary[0]);
    }
    void initialize_vector(Xbyak::CodeGenerator*          cg,
                           std::vector<Xbyak::Ymm> const& auxillary,
                           avx2_plus) override
    {
        assert(auxillary.size());
        auxillary_index = auxillary[0].getIdx();
        cg->vxorpd(auxillary[0], auxillary[0], auxillary[0]);
    }
    void initialize_vector(Xbyak::CodeGenerator*          cg,
                           std::vector<Xbyak::Ymm> const& auxillary,
                           avx2) override
    {
        assert(auxillary.size());
        auxillary_index = auxillary[0].getIdx();
        cg->vxorpd(auxillary[0], auxillary[0], auxillary[0]);
    }

    void initialize_scalar(Xbyak::CodeGenerator*,
                           std::vector<Xbyak::Xmm> const&, avx512) override
    {
    }
    void initialize_scalar(Xbyak::CodeGenerator*,
                           std::vector<Xbyak::Xmm> const&, avx2_plus) override
    {
    }
    void initialize_scalar(Xbyak::CodeGenerator*,
                           std::vector<Xbyak::Xmm> const&, avx2) override
    {
    }

    void process_vector(Xbyak::CodeGenerator* cg, Xbyak::Zmm const& dest,
                        std::vector<Xbyak::Zmm> const&, avx512) override
    {
        assert(auxillary_index >= 0 && auxillary_index < 32);
        cg->vmaxps(dest, dest, Xbyak::Zmm(auxillary_index));
    }
    void process_vector(Xbyak::CodeGenerator* cg, Xbyak::Ymm const& dest,
                        std::vector<Xbyak::Ymm> const&, avx2_plus) override
    {
        assert(auxillary_index >= 0 && auxillary_index < 16);
        cg->vmaxps(dest, dest, Xbyak::Ymm(auxillary_index));
    }
    void process_vector(Xbyak::CodeGenerator* cg, Xbyak::Ymm const& dest,
                        std::vector<Xbyak::Ymm> const&, avx2) override
    {
        assert(auxillary_index >= 0 && auxillary_index < 16);
        cg->vmaxps(dest, dest, Xbyak::Ymm(auxillary_index));
    }

    void process_scalar(Xbyak::CodeGenerator* cg, Xbyak::Xmm const& dest,
                        std::vector<Xbyak::Xmm> const& auxillary,
                        avx512) override
    {
        assert(auxillary.size() > 0);
        assert(auxillary[0].getIdx() < 16);
        cg->xorpd(auxillary[0], auxillary[0]);
        cg->maxps(dest, auxillary[0]);
    }
    void process_scalar(Xbyak::CodeGenerator* cg, Xbyak::Xmm const& dest,
                        std::vector<Xbyak::Xmm> const& auxillary,
                        avx2_plus) override
    {
        process_scalar(cg, dest, auxillary, avx512());
    }
    void process_scalar(Xbyak::CodeGenerator* cg, Xbyak::Xmm const& dest,
                        std::vector<Xbyak::Xmm> const& auxillary, avx2) override
    {
        process_scalar(cg, dest, auxillary, avx512());
    }
};

} // namespace aot
} // namespace sysml
} // namespace facebook
