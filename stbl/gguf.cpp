#include "gguf.h"

#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <stdexcept>

namespace {

enum : uint32_t { T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64, T_I64, T_F64 };

uint64_t scalar_size(uint32_t t) {
    switch (t) {
    case T_U8: case T_I8: case T_BOOL: return 1;
    case T_U16: case T_I16: return 2;
    case T_U32: case T_I32: case T_F32: return 4;
    case T_U64: case T_I64: case T_F64: return 8;
    }
    throw std::runtime_error("bad GGUF value type");
}

struct Reader {
    const uint8_t *p, *end;
    template <class T> T get() {
        if (p + sizeof(T) > end) throw std::runtime_error("truncated GGUF header");
        T v;
        memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }
    std::string str() {
        const uint64_t n = get<uint64_t>();
        if (p + n > end) throw std::runtime_error("truncated GGUF string");
        std::string s((const char *)p, n);
        p += n;
        return s;
    }
    void skip(uint32_t t) {
        if (t == T_STR) { str(); return; }
        if (t == T_ARR) {
            const uint32_t et = get<uint32_t>();
            const uint64_t n = get<uint64_t>();
            if (et == T_STR || et == T_ARR) { for (uint64_t i = 0; i < n; i++) skip(et); return; }
            p += n * scalar_size(et);
            return;
        }
        p += scalar_size(t);
    }
};

uint64_t type_nbytes(uint32_t type, uint64_t n) {
    switch (type) {
    case QT_F32: return n * 4;
    case QT_BF16: return n * 2;
    case QT_I8: case QT_STBL_E4M3: case QT_STBL_E8M0: return n;
    case QT_MXFP4: return n / 32 * 17;
    }
    throw std::runtime_error("unsupported tensor type " + std::to_string(type));
}

}  // namespace

bool GgufFile::open(const char *path, std::string &err) {
    try {
        fd = ::open(path, O_RDONLY);
        if (fd < 0) throw std::runtime_error(std::string("cannot open ") + path);
        struct stat st;
        fstat(fd, &st);
        size = st.st_size;
        void *m = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
        if (m == MAP_FAILED) throw std::runtime_error("mmap failed");
        map = (const uint8_t *)m;
        Reader r{map, map + size};
        if (r.get<uint32_t>() != 0x46554747u) throw std::runtime_error("not a GGUF file");
        r.get<uint32_t>();
        const uint64_t n_tensors = r.get<uint64_t>(), n_kv = r.get<uint64_t>();
        uint64_t alignment = 32;
        for (uint64_t i = 0; i < n_kv; i++) {
            std::string key = r.str();
            const uint32_t t = r.get<uint32_t>();
            kv[key] = {t, (uint64_t)(r.p - map)};
            if (key == "general.alignment") alignment = *(const uint32_t *)r.p;
            r.skip(t);
        }
        std::vector<GgufTensor> list;
        for (uint64_t i = 0; i < n_tensors; i++) {
            GgufTensor t;
            t.name = r.str();
            const uint32_t nd = r.get<uint32_t>();
            uint64_t n = 1;
            for (uint32_t d = 0; d < nd; d++) { t.dims.push_back(r.get<uint64_t>()); n *= t.dims.back(); }
            t.type = r.get<uint32_t>();
            t.offset = r.get<uint64_t>();
            t.nbytes = type_nbytes(t.type, n);
            list.push_back(t);
        }
        const uint64_t data = ((uint64_t)(r.p - map) + alignment - 1) / alignment * alignment;
        for (auto &t : list) {
            t.offset += data;
            if (t.offset + t.nbytes > size) throw std::runtime_error("tensor " + t.name + " past end of file");
            tensors[t.name] = t;
        }
        return true;
    } catch (const std::exception &e) {
        err = e.what();
        return false;
    }
}

GgufFile::~GgufFile() {
    if (map) munmap((void *)map, size);
    if (fd >= 0) close(fd);
}

const GgufTensor *GgufFile::tensor(const std::string &name) const {
    auto it = tensors.find(name);
    return it == tensors.end() ? nullptr : &it->second;
}

uint32_t GgufFile::u32(const std::string &key) const {
    auto it = kv.find(key);
    if (it == kv.end() || it->second.first != T_U32) throw std::runtime_error("missing u32 " + key);
    return *(const uint32_t *)(map + it->second.second);
}

float GgufFile::f32(const std::string &key) const {
    auto it = kv.find(key);
    if (it == kv.end() || it->second.first != T_F32) throw std::runtime_error("missing f32 " + key);
    return *(const float *)(map + it->second.second);
}

std::vector<uint64_t> GgufFile::uints(const std::string &key) const {
    auto it = kv.find(key);
    if (it == kv.end() || it->second.first != T_ARR) throw std::runtime_error("missing array " + key);
    Reader r{map + it->second.second, map + size};
    const uint32_t et = r.get<uint32_t>();
    const uint64_t n = r.get<uint64_t>();
    std::vector<uint64_t> v(n);
    for (uint64_t i = 0; i < n; i++) {
        if (et == T_U32) v[i] = r.get<uint32_t>();
        else if (et == T_U64) v[i] = r.get<uint64_t>();
        else throw std::runtime_error("array " + key + " is not u32/u64");
    }
    return v;
}
