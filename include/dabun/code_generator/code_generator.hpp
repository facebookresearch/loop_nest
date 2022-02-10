// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "dabun/code_generator/aot_fn.hpp"
#include "dabun/code_generator/memory_resource.hpp"
#include "dabun/code_generator/xbyak.hpp"

#include "dabun/core.hpp"
#include "dabun/isa.hpp"

#include <any>
#include <cassert>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <vector>

namespace dabun
{

class xbyak_allocator_adapter : public Xbyak::Allocator
{
private:
    memory_resource* resource_;
    std::set<void*>  managed_;

public:
    xbyak_allocator_adapter(memory_resource* resource)
        : resource_(resource)
    {
    }

    ~xbyak_allocator_adapter() { assert(managed_.empty()); }

    xbyak_buffer_type* alloc(std::size_t size) final override
    {
        auto ptr = resource_->allocate_bytes(size * sizeof(xbyak_buffer_type));
        if (!resource_->is_inplace())
        {
            managed_.insert(ptr);
        }
        return reinterpret_cast<xbyak_buffer_type*>(ptr);
    }

    void free(xbyak_buffer_type* ptr) final override
    {
        if (!resource_->is_inplace())
        {
            auto it = managed_.find(ptr);
            if (it != managed_.end())
            {
                resource_->deallocate_bytes(ptr);
                managed_.erase(ptr);
            }
        }
    }

    bool useProtect() const final override { return false; }

    xbyak_buffer_type* release(xbyak_buffer_type* ptr)
    {
        if (!resource_->is_inplace())
        {
            auto it = managed_.find(ptr);
            if (it == managed_.end())
            {
                throw std::invalid_argument(
                    "pointer not managed by the allocator");
            }
            managed_.erase(it);
        }
        return ptr;
    }

    Xbyak::Allocator* self() { return this; }

    auto get_deleter()
    {
        assert(!resource_->is_inplace());
        return [resource = this->resource_](xbyak_buffer_type* ptr) {
            resource->deallocate_bytes(ptr);
        };
    }

    bool is_inplace() const { return resource_->is_inplace(); }
};

class basic_code_generator : private xbyak_allocator_adapter,
                             public Xbyak::CodeGenerator
{
private:
    template <class T>
    T get_unique_or_shared()
    {
        assert(!xbyak_allocator_adapter::is_inplace());
        ready();
        std::size_t size = getSize() * sizeof(xbyak_buffer_type);
        auto        ptr  = xbyak_allocator_adapter::release(
            const_cast<xbyak_buffer_type*>(getCode()));
        return T(ptr, size, get_deleter());
    }

#if defined(DABUN_ARCH_AARCH64)

protected:
    std::vector<std::any> raii;

public:
    std::shared_ptr<Xbyak::Label> make_label()
    {
        auto ret = std::make_shared<Xbyak::Label>();
        raii.push_back(ret);
        return ret;
    }

#endif

public:
#if !defined(DABUN_ARCH_AARCH64)
    using Reg64  = Xbyak::Reg64;
    using Label  = Xbyak::Label;
    using Xmm    = Xbyak::Xmm;
    using Ymm    = Xbyak::Ymm;
    using Zmm    = Xbyak::Zmm;
    using OpMask = Xbyak::Opmask;

    auto argument_address(std::size_t N) const
    {
        assert(N > 5);
        return ptr[rsp + (N - 5 * 8)];
    }

    void align_to(unsigned alignment)
    {
        if (getSize() % alignment)
        {
            nop(alignment - static_cast<unsigned>(getSize() % alignment));
        }
    }

#else

    using Reg64 = Xbyak::XReg;
    using Label = Xbyak::LabelAArch64;
    using VReg  = Xbyak::VReg;
    using HReg  = Xbyak::HReg;
    using SReg  = Xbyak::SReg;
    using DReg  = Xbyak::DReg;
    using QReg  = Xbyak::QReg;
    using WReg  = Xbyak::WReg;
    using XReg  = Xbyak::XReg;

#endif

    explicit basic_code_generator(
        memory_resource* resource = memory_resource::default_resource())
        : xbyak_allocator_adapter(resource)
        , Xbyak::CodeGenerator(65536, Xbyak::AutoGrow,
                               xbyak_allocator_adapter::self())
    {
    }

    template <class Signature>
    unique_aot_fn<Signature> get_unique_fn() &&
    {
        return get_unique_or_shared<unique_aot_fn<Signature>>();
    }

    template <class Signature>
    shared_aot_fn<Signature> get_shared_fn() &&
    {
        return get_unique_or_shared<shared_aot_fn<Signature>>();
    }

    template <class Signature>
    aot_fn_ref<Signature> get_reference_fn() &&
    {
        assert(xbyak_allocator_adapter::is_inplace());
        ready();
        std::size_t size = getSize() * sizeof(xbyak_buffer_type);
        auto        ptr  = const_cast<xbyak_buffer_type*>(getCode());
        return aot_fn_ref<Signature>(ptr, size);
    }
};

template <class>
class code_generator;

template <class Ret, class... Args>
class code_generator<Ret(Args...)> : public basic_code_generator
{
private:
    using Signature = identity_type_t<Ret(Args...)>;

public:
    explicit code_generator(
        memory_resource* resource = memory_resource::default_resource())
        : basic_code_generator(resource)
    {
    }

    unique_aot_fn<Signature> get_unique() &&
    {
        basic_code_generator* base = this;
        return std::move(*base).template get_unique_fn<Signature>();
    }

    shared_aot_fn<Signature> get_shared() &&
    {
        basic_code_generator* base = this;
        return std::move(*base).template get_shared_fn<Signature>();
    }

    aot_fn_ref<Signature> get_reference() &&
    {
        basic_code_generator* base = this;
        return std::move(*base).template get_reference_fn<Signature>();
    }
};

template <class, class>
class with_signature;

template <class BasicCodeGenerator, class Ret, class... Args>
class with_signature<BasicCodeGenerator, Ret(Args...)>
{
private:
    using Signature = identity_type_t<Ret(Args...)>;

    BasicCodeGenerator& self()
    {
        static_assert(
            std::is_base_of_v<basic_code_generator, BasicCodeGenerator>);
        return static_cast<BasicCodeGenerator&>(*this);
    }

public:
    unique_aot_fn<Signature> get_unique() &&
    {
        return std::move(self()).template get_unique_fn<Signature>();
    }

    shared_aot_fn<Signature> get_shared() &&
    {
        return std::move(self()).template get_shared_fn<Signature>();
    }

    aot_fn_ref<Signature> get_reference() &&
    {
        return std::move(self()).template get_reference_fn<Signature>();
    }
};

} // namespace dabun
