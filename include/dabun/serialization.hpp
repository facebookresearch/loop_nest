#pragma once

#include "dabun/loop_nest.hpp"

#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <dirent.h>

namespace dabun
{
namespace detail
{

static inline std::string to_string(std::string s) { return s; }

static inline std::string to_string(int v) { return std::to_string(v); }

template <class K, class V>
static inline std::string to_string(std::map<K, V> m)
{
    std::ostringstream out;
    int                i = 1;
    int                n = m.size();
    for (auto const& kv : m)
    {
        out << to_string(kv.first);
        out << ",";
        out << to_string(kv.second);
        if (i < n)
        {
            out << ",";
        }
        i += 1;
    }
    return out.str();
}

template <class V>
static inline std::string to_string(const std::set<V> s)
{
    std::ostringstream out;
    int                i = 1;
    int                n = s.size();
    for (auto const& e : s)
    {
        out << to_string(e);
        if (i < n)
        {
            out << ",";
        }
        i += 1;
    }
    return out.str();
}

template <class V1, class V2>
static inline std::string to_string(const std::pair<V1, V2> p)
{
    std::ostringstream out;
    out << to_string(p.first);
    out << ",";
    out << to_string(p.second);
    return out.str();
}

static inline std::vector<std::string> get_tokens(std::string line, char delim)
{
    std::vector<std::string> tokens;
    std::stringstream        ss(line);
    while (ss.good())
    {
        std::string substr;
        std::getline(ss, substr, delim);
        tokens.push_back(substr);
    }
    return tokens;
}

static inline std::vector<std::vector<std::string>>
get_tokenized(std::string str)
{
    auto lines = get_tokens(str, '\n');

    std::vector<std::vector<std::string>> tokenized;
    for (auto const& line : lines)
    {
        std::vector<std::string> tokens = get_tokens(line, ',');
        if (tokens.size() > 0)
        {
            tokenized.push_back(tokens);
        }
    }

    return tokenized;
}

inline void initialize_serialized_ct(std::string output_dir,
                                     int&        serialized_file_ct)
{
    int  i    = 0;
    DIR* dirp = opendir(output_dir.c_str());
    if (dirp == NULL)
    {
        throw std::runtime_error(
            "loop_nest serialization failed: No such directory " + output_dir);
    }
    dirent* dp;

    while ((dp = readdir(dirp)) != NULL)
    {
        std::string file_name(dp->d_name);
        if (file_name != "." && file_name != "..")
        {
            i++;
        }
    }
    closedir(dirp);
    serialized_file_ct = i;
}

inline std::string get_file_path(std::string output_dir,
                                 std::string suffix = ".txt")
{
    static int serialized_file_ct = -1;

    if (serialized_file_ct < 0)
    {
        initialize_serialized_ct(output_dir, serialized_file_ct);
    }

    std::ostringstream file_path_ss;
    file_path_ss << output_dir;
    file_path_ss << "/";
    file_path_ss << std::to_string(++serialized_file_ct);
    file_path_ss << suffix;
    return file_path_ss.str();
}

class serialized_loop_nest_inputs
{

private:
    std::vector<std::pair<std::string, int>>          order;
    std::map<std::string, int>                        sizes;
    std::map<std::string, std::set<std::string>>      formulas;
    std::map<std::string, std::map<std::string, int>> strides;
    // assumes FMA
    std::optional<int> user_unroll_limit;
    // assumes no pre/post-ops for now
    // these are kind of janky in the halide translator...

    std::string order_to_string()
    {
        std::ostringstream out;
        int                i = 1;
        int                n = order.size();
        for (auto const& kv : order)
        {
            out << to_string(kv);
            if (i < n)
            {
                out << ",";
            }
            i += 1;
        }
        return out.str();
    }

    std::string sizes_to_string()
    {
        std::ostringstream out;
        out << to_string(sizes);
        return out.str();
    }

    std::string formulas_to_string()
    {
        std::ostringstream out;
        out << to_string(formulas.at("C")) << "\n";
        out << to_string(formulas.at("A")) << "\n";
        out << to_string(formulas.at("B"));
        return out.str();
    }

    std::string strides_to_string()
    {
        std::ostringstream out;
        out << to_string(strides.at("C")) << "\n";
        out << to_string(strides.at("A")) << "\n";
        out << to_string(strides.at("B"));
        return out.str();
    }

    std::string unroll_limit_to_string()
    {
        std::ostringstream out;
        if (user_unroll_limit)
        {
            out << std::to_string(*user_unroll_limit);
        }
        else
        {
            out << "null";
        }
        return out.str();
    }

    std::vector<std::pair<std::string, int>>
    parse_vector_pairs(std::vector<std::string> tokens)
    {
        std::vector<std::pair<std::string, int>> out;
        std::pair<std::string, int>              entry;
        for (int i = 0; i < tokens.size(); i++)
        {
            if (i % 2 == 0)
            {
                entry = {tokens[i], -1};
            }
            else
            {
                entry = {entry.first, std::stoi(tokens[i])};
                out.push_back(entry);
            }
        }
        return out;
    }

    std::set<std::string> parse_set(std::vector<std::string> tokens)
    {
        std::set<std::string> formula(tokens.begin(), tokens.end());
        return formula;
    }

    std::map<std::string, int> parse_map(std::vector<std::string> tokens)
    {
        std::map<std::string, int> out;
        std::string                key;
        for (int i = 0; i < tokens.size(); i++)
        {
            if (i % 2 == 0)
            {
                key = tokens[i];
            }
            else
            {
                out[key] = std::stoi(tokens[i]);
            }
        }
        return out;
    }

public:
    std::vector<std::pair<std::string, int>> get_order() const { return order; }

    std::set<std::string> get_formula(std::string name) const
    {
        return formulas.at(name);
    }

    std::map<std::string, int> get_sizes() const { return sizes; }

    std::map<std::string, int> get_strides(std::string name) const
    {
        return strides.at(name);
    }

    std::optional<int> get_unroll_limit() const { return user_unroll_limit; }

    // loop nest style constructor
    serialized_loop_nest_inputs(
        std::vector<std::pair<std::string, int>> const& order,
        std::map<std::string, int> const&               sizes,
        std::set<std::string> const&                    C_formula,
        std::set<std::string> const&                    A_formula,
        std::set<std::string> const&                    B_formula,
        std::map<std::string, int> const&               C_strides,
        std::map<std::string, int> const&               A_strides,
        std::map<std::string, int> const&               B_strides,
        std::optional<int> user_fma_unroll_limit = std::nullopt)
        : order(order)
        , sizes(sizes)
        , formulas({{"C", C_formula}, {"A", A_formula}, {"B", B_formula}})
        , strides({{"C", C_strides}, {"A", A_strides}, {"B", B_strides}})
        , user_unroll_limit(user_fma_unroll_limit)
    {
    }
    // serialize
    std::string str()
    {
        std::stringstream out;
        out << order_to_string() << "\n";
        out << sizes_to_string() << "\n";
        out << formulas_to_string() << "\n";
        out << strides_to_string() << "\n";
        out << unroll_limit_to_string();
        return out.str();
    }

    void to_file(std::string file_path)
    {

        std::ofstream fout(file_path);
        fout << str();
        fout.close();
    }

    // deserialize
    serialized_loop_nest_inputs(std::string input)
    {
        auto tokenized_lines = get_tokenized(input);
        assert(tokenized_lines.size() == 9);

        auto order     = parse_vector_pairs(tokenized_lines.at(0));
        auto sizes     = parse_map(tokenized_lines.at(1));
        auto C_formula = parse_set(tokenized_lines.at(2));
        auto A_formula = parse_set(tokenized_lines.at(3));
        auto B_formula = parse_set(tokenized_lines.at(4));
        auto C_strides = parse_map(tokenized_lines.at(5));
        auto A_strides = parse_map(tokenized_lines.at(6));
        auto B_strides = parse_map(tokenized_lines.at(7));

        std::string        unroll_limit_str = tokenized_lines.at(8).at(0);
        std::optional<int> user_unroll_limit;
        if (unroll_limit_str == "null")
        {
            user_unroll_limit = std::nullopt;
        }
        else
        {
            user_unroll_limit =
                std::make_optional<int>(std::stoi(unroll_limit_str));
        }

        this->order    = order;
        this->sizes    = sizes;
        this->formulas = {{"C", C_formula}, {"A", A_formula}, {"B", B_formula}};
        this->strides  = {{"C", C_strides}, {"A", A_strides}, {"B", B_strides}};
        this->user_unroll_limit = user_unroll_limit;
    }

    static serialized_loop_nest_inputs from_str(std::string str)
    {
        return serialized_loop_nest_inputs(str);
    }

    static serialized_loop_nest_inputs from_file(std::string filename)
    {
        std::ifstream t(filename);
        std::string   str((std::istreambuf_iterator<char>(t)),
                        std::istreambuf_iterator<char>());
        return serialized_loop_nest_inputs::from_str(str);
    }
};

} // namespace detail

using detail::serialized_loop_nest_inputs;

inline void
save_loop_nest_inputs(const std::string& output_dir,
                      const std::vector<std::pair<std::string, int>>& order,
                      const std::map<std::string, int>&               sizes,
                      const std::set<std::string>&                    C_formula,
                      const std::set<std::string>&                    A_formula,
                      const std::set<std::string>&                    B_formula,
                      const std::map<std::string, int>&               C_strides,
                      const std::map<std::string, int>&               A_strides,
                      const std::map<std::string, int>&               B_strides,
                      std::optional<int> unroll_limit)
{
    auto serializer = detail::serialized_loop_nest_inputs(
        order, sizes, C_formula, A_formula, B_formula, C_strides, A_strides,
        B_strides, unroll_limit);
    auto file_path = detail::get_file_path(output_dir);
    serializer.to_file(file_path);
}

} // namespace dabun
