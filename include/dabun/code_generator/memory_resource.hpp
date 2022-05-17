// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "dabun/hask/aligned_alloc.hpp"

#include <cstddef>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <unordered_map>

#if defined(__GNUC__)
#    include <sys/mman.h>
#endif

#if defined(__APPLE__)
#    include "dabun/hask/apple.hpp"
#endif

namespace dabun
{

// Forward declarations
class memory_resource;
class malloc_memory_resource;
class mmap_memory_resource;
class inplace_memory_resource;

class memory_resource
{
public:
    memory_resource() {}
    memory_resource(memory_resource const&) = delete;
    memory_resource& operator=(memory_resource const&) = delete;

    void* allocate_bytes(std::size_t size)
    {
        return this->do_allocate_bytes(size);
    }
    void deallocate_bytes(void* ptr) { this->do_deallocate_bytes(ptr); }

    virtual ~memory_resource() {}
    virtual void* do_allocate_bytes(std::size_t size) = 0;
    virtual void  do_deallocate_bytes(void* ptr)      = 0;
    virtual bool  is_inplace() const                  = 0;

    static memory_resource* default_resource();
};

class malloc_memory_resource : public memory_resource
{
public:
    void* do_allocate_bytes(std::size_t size) final override
    {
        constexpr std::size_t ALIGN_PAGE_SIZE = 4096;
        return hask::checked_aligned_alloc(ALIGN_PAGE_SIZE, size);
    }

    void do_deallocate_bytes(void* ptr) final override { std::free(ptr); }

    bool is_inplace() const final override { return false; }
};

#if defined(__GNUC__)

class mmap_memory_resource : public memory_resource
{
private:
    std::unordered_map<void*, std::size_t> sizes_;

public:
    void* do_allocate_bytes(std::size_t size) final override
    {
        static constexpr size_t ALIGN_PAGE_SIZE = 4096;

        std::size_t const aligned_mask = ALIGN_PAGE_SIZE - 1;
        size                           = (size + aligned_mask) & ~aligned_mask;

#    if defined(__APPLE__)
        int const mode =
            MAP_PRIVATE | MAP_ANONYMOUS |
            ((hask::get_macOS_version() >= hask::mojave_version ? MAP_JIT : 0));
#    else
        int const mode = MAP_PRIVATE | MAP_ANONYMOUS;
#    endif

        void* ptr = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, mode, -1, 0);

        if (ptr == MAP_FAILED || ptr == nullptr)
        {
            throw std::bad_alloc();
        }

        sizes_[ptr] = size;
        return ptr;
    }

    void do_deallocate_bytes(void* ptr) final override
    {
        if (ptr == nullptr)
        {
            return;
        }

        auto it = sizes_.find(ptr);

        if (it == sizes_.end())
        {
            throw std::invalid_argument(
                "Pointer was not allocated with mmap_memory_resource");
        }

        munmap(it->first, it->second);
        sizes_.erase(it);
    }

    bool is_inplace() const final override { return false; }
};

#endif

class inplace_memory_resource : public memory_resource
{
private:
    void*       memory_;
    std::size_t size_;

public:
    inplace_memory_resource(void* memory, std::size_t size)
        : memory_(memory)
        , size_(size)
    {
    }

    void* do_allocate_bytes(std::size_t size) final override
    {
        if (size > size_)
        {
            throw std::bad_alloc();
        }
        return memory_;
    }

    void do_deallocate_bytes(void*) final override
    { // no op
    }

    bool is_inplace() const final override { return true; }
};

inline memory_resource* memory_resource::default_resource()
{
    static malloc_memory_resource malloc_resource;

#if defined(__APPLE__)
    static mmap_memory_resource mmap_resource;
    if (hask::get_macOS_version() >= hask::mojave_version)
    {
        return &mmap_resource;
    }
#endif

    return &malloc_resource;
}

} // namespace dabun
