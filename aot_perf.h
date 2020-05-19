// Parts adopted from
// github.com/facebook/hhvm/blob/master/hphp/runtime/vm/debug/perf-jitdump.cpp

/*
   +----------------------------------------------------------------------+
   | HipHop for PHP                                                       |
   +----------------------------------------------------------------------+
   | Copyright (c) 2010-present Facebook, Inc. (http://www.facebook.com)  |
   +----------------------------------------------------------------------+
   | This source file is subject to version 3.01 of the PHP license,      |
   | that is bundled with this package in the file LICENSE, and is        |
   | available through the world-wide-web at the following url:           |
   | http://www.php.net/license/3_01.txt                                  |
   | If you did not receive a copy of the PHP license and are unable to   |
   | obtain it through the world-wide-web, please send a note to          |
   | license@php.net so we can mail you a copy immediately.               |
   +----------------------------------------------------------------------+

 * perf-jitdump.c: perf agent to dump generated code into jit-<pid>.dump
 *
 * Adapted from the Oprofile code in opagent.c
 * Copyright 2007 OProfile authors
 * Jens Wilke
 * Daniel Hansel
 * Copyright IBM Corporation 2007
 */

#pragma once

#define XBYAK_NO_OP_NAMES
#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"

#include "log.h"
#include "oprof-jitdump.h"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>

#include <fcntl.h>
#include <sys/types.h>
#include <time.h>

namespace facebook
{
namespace sysml
{
namespace aot
{

namespace detail
{
class profiler_wrapper
{
private:
    Xbyak::util::Profiler profiler_;

    bool use_arch_timestamp;

#if defined(CLOCK_MONOTONIC)
    clockid_t perf_clk_id = CLOCK_MONOTONIC;
#else
    clockid_t perf_clk_id = CLOCK_REALTIME;
#endif

    FILE*       m_perfJitDump = nullptr;
    std::string m_perfJitDumpName;
    void*       m_perfMmapMarker = nullptr;

    // From
    // github.com/facebook/hhvm/blob/master/hphp/util/cycles.h
    // (same licence as above)

    /*
     * Return the underlying machine cycle counter. While this is slightly
     * non-portable in theory, all the CPUs you're likely to care about support
     * it in some way or another.
     */
    inline uint64_t cpuCycles()
    {
#ifdef __x86_64__
        uint64_t lo, hi;
        asm volatile("rdtsc" : "=a"((lo)), "=d"(hi));
        return lo | (hi << 32);
#elif __powerpc64__
        // This returns a time-base
        uint64_t tb;
        asm volatile("mfspr %0, 268" : "=r"(tb));
        return tb;
#elif _MSC_VER
        return (uint64_t)__rdtsc();
#elif __aarch64__
        // FIXME: This returns the virtual timer which is not exactly
        // the core cycles but has a different frequency.
        uint64_t tb;
        asm volatile("mrs %0, cntvct_el0" : "=r"(tb));
        return tb;
#else
        not_implemented();
#endif
    }
    inline uint64_t perfGetTimestamp(void)
    {
        struct timespec ts;

        /* cpuCycles returns arch TS, rdtsc value */
        if (use_arch_timestamp)
            return cpuCycles();

        if (clock_gettime(perf_clk_id, &ts))
            return 0;

        return timespec_to_ns(&ts);
    }

    static inline uint64_t timespec_to_ns(const struct timespec* ts)
    {
        return ((uint64_t)ts->tv_sec * 1000000000) + ts->tv_nsec;
    }

    static int getEMachine(JitHeader* /*hdr*/)
    {
        char id[16];
        int  fd;
        struct
        {
            uint16_t e_type;
            uint16_t e_machine;
        } info;

        fd = open("/proc/self/exe", O_RDONLY);
        if (fd == -1)
            return -1;
        read(fd, id, sizeof(id));

        /* check ELF signature */
        if (id[0] != 0x7f || id[1] != 'E' || id[2] != 'L' || id[3] != 'F')
        {
            close(fd);
            return -1;
        }

        read(fd, &info, sizeof(info));
        close(fd);
        return 0;
    }

    void initPerfJitDump()
    {
        {
            int const pid = getpid();
            char      name[512];
            snprintf(name, sizeof(name), "/tmp/jit-%d.dump", pid);
            m_perfJitDumpName = name;
        }

        LN_LOG(DEBUG) << "Jitdump file: " << m_perfJitDumpName << "\n";

        use_arch_timestamp = false; // true

        const char* str = getenv("JITDUMP_USE_ARCH_TIMESTAMP");
        if (str && *str == '1')
        {
            use_arch_timestamp = true;
        }

        /* check if perf_clk_id is supported else exit
         * jitdump records need to be timestamp'd
         */
        if (!perfGetTimestamp())
        {
            if (use_arch_timestamp)
            {
                LN_LOG(DEBUG) << "system arch timestamp not supported\n";
            }
            else
            {
                LN_LOG(DEBUG) << "kernel does not support (monotonic) "
                              << perf_clk_id << " clk_id\n";
            }
            LN_LOG(DEBUG) << "Cannot create " << m_perfJitDumpName << "\n";
            return;
        }

        int fd =
            open(m_perfJitDumpName.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0666);
        if (fd < 0)
        {
            LN_LOG(DEBUG) << "Failed to create the file " << m_perfJitDumpName
                          << " for perf jit dump: " << strerror(errno) << "\n";
            return;
        }

        m_perfMmapMarker = mmap(nullptr, sysconf(_SC_PAGESIZE),
                                PROT_READ | PROT_EXEC, MAP_PRIVATE, fd, 0);

        if (m_perfMmapMarker == MAP_FAILED)
        {
            LN_LOG(DEBUG) << "Failed to create mmap marker file for perf"
                          << "\n";
            close(fd);
            return;
        }

        m_perfJitDump = fdopen(fd, "w+");

        /* Init the jitdump header and write to the file */
        JitHeader header{};

        if (getEMachine(&header))
        {
            LN_LOG(DEBUG) << "failed to get machine ELF information"
                          << "\n";
            fclose(m_perfJitDump);
        }

        header.magic      = JITHEADER_MAGIC;
        header.version    = JITHEADER_VERSION;
        header.total_size = sizeof(header);
        header.pid        = getpid();

        int padding_count;
        /* calculate amount of padding '\0' */
        padding_count = PADDING_8ALIGNED(header.total_size);
        header.total_size += padding_count;

        header.timestamp = perfGetTimestamp();

        if (use_arch_timestamp)
        {
            header.flags |= JITDUMP_FLAGS_ARCH_TIMESTAMP;
        }

        fwrite(&header, sizeof(header), 1, m_perfJitDump);

        char const padding_bytes[7] = {'\0', '\0', '\0', '\0',
                                       '\0', '\0', '\0'};

        /* write padding '\0' if necessary */
        if (padding_count)
        {
            fwrite(padding_bytes, padding_count, 1, m_perfJitDump);
        }

        fflush(m_perfJitDump);
    }

    void closePerfJitDump()
    {
        if (!m_perfJitDump)
            return;

        JitRecCodeClose rec;

        rec.p.id         = JitRecordType::JIT_CODE_CLOSE;
        rec.p.total_size = sizeof(rec);
        rec.p.timestamp  = perfGetTimestamp();

        fwrite(&rec, sizeof(rec), 1, m_perfJitDump);
        fflush(m_perfJitDump);
        fclose(m_perfJitDump);

        m_perfJitDump = nullptr;
        if (m_perfMmapMarker != MAP_FAILED)
        {
            munmap(m_perfMmapMarker, sysconf(_SC_PAGESIZE));
        }
    }

    int perfJitDumpTrace(const void* startAddr, const unsigned int size,
                         const char* symName)
    {

        if (!startAddr || !size)
            return -1;

        static int     code_generation = 1;
        JitRecCodeLoad rec;
        size_t         padding_count;

        uint64_t vma = reinterpret_cast<uintptr_t>(startAddr);

        rec.p.id         = JitRecordType::JIT_CODE_LOAD;
        rec.p.total_size = sizeof(rec) + strlen(symName) + 1;
        padding_count    = PADDING_8ALIGNED(rec.p.total_size);
        rec.p.total_size += padding_count;
        rec.p.timestamp = perfGetTimestamp();

        rec.code_size = size;
        rec.vma       = vma;
        rec.code_addr = vma;
        rec.pid       = getpid();
        rec.tid       = getpid();
        rec.p.total_size += size;

        rec.code_index = code_generation++;

        /* write the name of the function and the jitdump record  */
        fwrite(&rec, sizeof(rec), 1, m_perfJitDump);
        fwrite(symName, (strlen(symName) + 1), 1, m_perfJitDump);

        char const padding_bytes[7] = {'\0', '\0', '\0', '\0',
                                       '\0', '\0', '\0'};

        /* write padding '\0' if necessary */
        if (padding_count)
        {
            fwrite(padding_bytes, padding_count, 1, m_perfJitDump);
        }

        /* write the code generated for the tracelet */
        fwrite(startAddr, size, 1, m_perfJitDump);

        fflush(m_perfJitDump);
        return 0;
    }

public:
    profiler_wrapper()
    {
        profiler_.init(Xbyak::util::Profiler::Perf);
        profiler_.setNameSuffix("-jit");

        initPerfJitDump();
    }

    ~profiler_wrapper() { closePerfJitDump(); }

    profiler_wrapper(profiler_wrapper const&) = delete;
    profiler_wrapper& operator=(profiler_wrapper const&) = delete;

    // Xbyak::util::Profiler& get() { return profiler_; }

    void set(const char* funcName, const void* startAddr, size_t funcSize)
    {
        profiler_.set(funcName, startAddr, funcSize);

        char buf[512];
        snprintf(buf, sizeof(buf), "%s%s", funcName, "-jit");
        perfJitDumpTrace(startAddr, funcSize, buf);
    }
};
} // namespace detail

inline detail::profiler_wrapper& get_xbyak_profiler()
{
    static detail::profiler_wrapper p;
    return p;
}

} // namespace aot
} // namespace sysml
} // namespace facebook
