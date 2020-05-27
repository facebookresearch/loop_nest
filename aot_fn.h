#pragma once

#define XBYAK_NO_OP_NAMES
#include "xbyak/xbyak.h"

#include "aot_perf.h"

#include <cstdint>
#include <fstream>
#include <functional>
#include <memory>
#include <string>

namespace facebook
{
namespace sysml
{
namespace aot
{

// Forward decl
class basic_code_generator;

template <typename Signature>
class unique_aot_fn;

template <typename Signature>
class weak_aot_fn;

template <typename Signature>
class shared_aot_fn;

template <typename Signature> // To be used with an inplace allocator
class aot_fn_ref;             // that is manually managing the execution
                              // priveleges

struct aot_fn_cast_tag
{
};

template <class To, class From>
struct aot_fn_caster;

template <class Deleter>
class mprotect_deleter_wrapper
{
private:
    Deleter     deleter_;
    std::size_t size_;

public:
    mprotect_deleter_wrapper() {}
    explicit mprotect_deleter_wrapper(Deleter deleter, std::size_t size)
        : deleter_(deleter)
        , size_(size)
    {
    }

    mprotect_deleter_wrapper(mprotect_deleter_wrapper const&) = default;
    mprotect_deleter_wrapper&
    operator=(mprotect_deleter_wrapper const&) = default;

    void operator()(Xbyak::uint8* ptr) const
    {
        Xbyak::CodeArray::protect(ptr, size_, Xbyak::CodeArray::PROTECT_RW);
        deleter_(ptr);
    }
};

template <typename ReturnType, typename... Args>
class unique_aot_fn<ReturnType(Args...)>
{
public:
    using function_pointer_type = ReturnType (*)(Args...);

private:
    using deleter_type =
        mprotect_deleter_wrapper<std::function<void(Xbyak::uint8*)>>;

    std::unique_ptr<Xbyak::uint8, deleter_type> executable_buffer_;
    std::size_t                                 size_ = 0;

    friend class basic_code_generator;

    template <typename Deleter>
    unique_aot_fn(Xbyak::uint8* buffer, std::size_t size, Deleter deleter)
        : executable_buffer_(buffer, deleter_type(deleter, size))
        , size_(size)
    {
        Xbyak::CodeArray::protect(buffer, size, Xbyak::CodeArray::PROTECT_RWE);
    }

    template <class, class>
    friend struct aot_fn_caster;

    template <typename OtherSignature>
    unique_aot_fn(unique_aot_fn<OtherSignature>&& other, aot_fn_cast_tag)
    {
        executable_buffer_ = std::move(other.executable_buffer_);
        size_              = std::exchange(other.size_, 0);
    }

public:
    unique_aot_fn() noexcept {}

    unique_aot_fn(unique_aot_fn const&) = delete;
    unique_aot_fn& operator=(unique_aot_fn const&) = delete;

    unique_aot_fn(unique_aot_fn&& other) noexcept = default;
    unique_aot_fn& operator=(unique_aot_fn&& other) noexcept = default;

    function_pointer_type get() const
    {
        return reinterpret_cast<function_pointer_type>(
            executable_buffer_.get());
    }

    ReturnType operator()(Args... args) const { return this->get()(args...); }

    explicit operator bool() const noexcept
    {
        return executable_buffer_ && *executable_buffer_;
    }

    void save_to_file(std::string const& fname) const
    {
        std::ofstream fout(fname.c_str(), std::ios::out | std::ios::binary);
        fout.write(reinterpret_cast<char*>(executable_buffer_.get()), size_);
    }

    void register_perf(std::string const& name = "")
    {
        get_xbyak_profiler().set(name.c_str(), executable_buffer_.get(),
                                 (int)size_);
    }
};

template <typename ReturnType, typename... Args>
class shared_aot_fn<ReturnType(Args...)>
{
public:
    using function_pointer_type = ReturnType (*)(Args...);

private:
    std::shared_ptr<Xbyak::uint8> executable_buffer_;
    std::size_t                   size_ = 0;

    friend class weak_aot_fn<ReturnType(Args...)>;

    shared_aot_fn(std::weak_ptr<Xbyak::uint8> const& weak_buffer,
                  std::size_t                        size)
        : executable_buffer_(weak_buffer.lock())
        , size_(size)
    {
    }

    friend class basic_code_generator;

    template <typename Deleter>
    shared_aot_fn(Xbyak::uint8* buffer, std::size_t size, Deleter deleter)
        : executable_buffer_(buffer,
                             mprotect_deleter_wrapper<Deleter>(deleter, size))
        , size_(size)
    {
        Xbyak::CodeArray::protect(buffer, size, Xbyak::CodeArray::PROTECT_RWE);
    }

    template <class, class>
    friend struct aot_fn_caster;

    template <typename OtherSignature>
    shared_aot_fn(shared_aot_fn<OtherSignature>&& other, aot_fn_cast_tag)
    {
        executable_buffer_ = std::move(other.executable_buffer_);
        size_              = std::exchange(other.size_, 0);
    }

    template <typename OtherSignature>
    shared_aot_fn(shared_aot_fn<OtherSignature> const& other, aot_fn_cast_tag)
    {
        executable_buffer_ = other.executable_buffer_;
        size_              = other.size_;
    }

public:
    shared_aot_fn() noexcept {}

    shared_aot_fn(shared_aot_fn const& other) = default;
    shared_aot_fn& operator=(shared_aot_fn const& other) = default;
    shared_aot_fn(shared_aot_fn&& other)                 = default;
    shared_aot_fn& operator=(shared_aot_fn&& other) = default;

    function_pointer_type get() const
    {
        return reinterpret_cast<function_pointer_type>(
            executable_buffer_.get());
    }

    ReturnType operator()(Args... args) const { return this->get()(args...); }

    explicit operator bool() const noexcept
    {
        return executable_buffer_ && *executable_buffer_;
    }

    void save_to_file(std::string const& fname) const
    {
        std::ofstream fout(fname.c_str(), std::ios::out | std::ios::binary);
        fout.write(reinterpret_cast<char*>(executable_buffer_.get()), size_);
    }

    void register_perf(std::string const& name = "")
    {
        get_xbyak_profiler().set(name.c_str(), executable_buffer_.get(),
                                 (int)size_);
    }
};

template <typename ReturnType, typename... Args>
class weak_aot_fn<ReturnType(Args...)>
{
public:
    using function_pointer_type = ReturnType (*)(Args...);

private:
    std::weak_ptr<Xbyak::uint8> weak_buffer_;
    std::size_t                 size_ = 0;

    using matching_shared_fn = shared_aot_fn<ReturnType(Args...)>;

public:
    weak_aot_fn() noexcept {}

    weak_aot_fn(weak_aot_fn const& other) = default;
    weak_aot_fn& operator=(weak_aot_fn const& other) = default;
    weak_aot_fn(weak_aot_fn&& other)                 = default;
    weak_aot_fn& operator=(weak_aot_fn&& other) = default;

    weak_aot_fn(matching_shared_fn const& shared)
        : weak_buffer_(shared.executable_buffer_)
        , size_(shared.size_)
    {
    }

    weak_aot_fn& operator=(matching_shared_fn const& shared)
    {
        weak_buffer_ = shared.executable_buffer_;
        return *this;
    }

    matching_shared_fn lock() const noexcept
    {
        return matching_shared_fn(weak_buffer_, size_);
    }
};

template <typename ReturnType, typename... Args>
class aot_fn_ref<ReturnType(Args...)>
{
public:
    using function_pointer_type = ReturnType (*)(Args...);

private:
    Xbyak::uint8* executable_buffer_ptr_ = nullptr;
    std::size_t   size_                  = 0;

    friend class basic_code_generator;

    explicit aot_fn_ref(Xbyak::uint8* buffer, std::size_t size) noexcept
        : executable_buffer_ptr_(buffer)
        , size_(size)
    {
    }

    template <class, class>
    friend struct aot_fn_caster;

    template <typename OtherSignature>
    aot_fn_ref(aot_fn_ref<OtherSignature> const& other, aot_fn_cast_tag)
    {
        executable_buffer_ptr_ = other.executable_buffer_ptr_;
        size_                  = other.size_;
    }

public:
    aot_fn_ref() noexcept {}
    aot_fn_ref(aot_fn_ref const&) noexcept = default;
    aot_fn_ref& operator=(aot_fn_ref const&) noexcept = default;

    aot_fn_ref(aot_fn_ref&& other) noexcept = default;
    aot_fn_ref& operator=(aot_fn_ref&& other) noexcept = default;

    function_pointer_type get() const noexcept
    {
        return reinterpret_cast<function_pointer_type>(executable_buffer_ptr_);
    }

    ReturnType operator()(Args... args) const { return this->get()(args...); }

    explicit operator bool() const noexcept { return executable_buffer_ptr_; }

    void save_to_file(std::string const& fname) const
    {
        std::ofstream fout(fname.c_str(), std::ios::out | std::ios::binary);
        fout.write(reinterpret_cast<char*>(executable_buffer_ptr_), size_);
    }

    void register_perf(std::string const& name = "")
    {
        get_xbyak_profiler().set(name.c_str(), executable_buffer_ptr_,
                                 (int)size_);
    }
};

template <class RetTo, class... ArgsTo, class RetFrom, class... ArgsFrom>
struct aot_fn_caster<unique_aot_fn<RetTo(ArgsTo...)>,
                     unique_aot_fn<RetFrom(ArgsFrom...)>>
{
    unique_aot_fn<RetTo(ArgsTo...)>
    do_cast(unique_aot_fn<RetFrom, ArgsFrom...>&& other)
    {
        return unique_aot_fn<RetTo(ArgsTo...)>(std::move(other),
                                               aot_fn_cast_tag());
    }
};

template <class RetTo, class... ArgsTo, class RetFrom, class... ArgsFrom>
struct aot_fn_caster<shared_aot_fn<RetTo(ArgsTo...)>,
                     shared_aot_fn<RetFrom(ArgsFrom...)>>
{
    shared_aot_fn<RetTo(ArgsTo...)>
    do_cast(shared_aot_fn<RetFrom, ArgsFrom...>&& other)
    {
        return shared_aot_fn<RetTo(ArgsTo...)>(std::move(other),
                                               aot_fn_cast_tag());
    }

    shared_aot_fn<RetTo(ArgsTo...)>
    do_cast(shared_aot_fn<RetFrom, ArgsFrom...> const& other)
    {
        return shared_aot_fn<RetTo(ArgsTo...)>(other, aot_fn_cast_tag());
    }
};

template <class RetTo, class... ArgsTo, class RetFrom, class... ArgsFrom>
struct aot_fn_caster<aot_fn_ref<RetTo(ArgsTo...)>,
                     aot_fn_ref<RetFrom(ArgsFrom...)>>
{
    aot_fn_ref<RetTo(ArgsTo...)>
    do_cast(aot_fn_ref<RetFrom, ArgsFrom...> const& other)
    {
        return aot_fn_ref<RetTo(ArgsTo...)>(other, aot_fn_cast_tag());
    }
};

template <class To, class From>
To aot_fn_cast(From&& from)
{
    aot_fn_caster<To, From> caster;
    return caster(std::forward<From>(from));
}

} // namespace aot
} // namespace sysml
} // namespace facebook
