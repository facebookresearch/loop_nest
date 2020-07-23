

template <class ISA>
class TransposeNode : public ProgramNode<ISA>
{

private:
    std::string                                       input;
    std::string                                       output;
    std::map<std::string, std::map<std::string, int>> strides;

    static const node_kind   type = node_kind::transpose_type;
    std::vector<std::string> tensors_used;

public:
    TransposeNode(std::string input, std::string output,
                  std::map<std::string, std::map<std::string, int>> strides)
        : input(input)
        , output(output)
        , strides(strides)
    {
        tensors_used.push_back(input);
        tensors_used.push_back(output);
    }

    std::vector<std::string> const& get_tensors_used() const
    {
        return tensors_used;
    }

    std::map<std::string, std::map<std::string, int>> const&
    get_tensor_strides() const
    {
        return strides;
    }

    void set_limits(std::map<std::string, std::vector<int>>)
    {
        // do nothing...
    }

    std::vector<std::shared_ptr<ProgramNode<ISA>>> get_children() const
    {
        return {};
    }

    void set_children(std::vector<std::shared_ptr<ProgramNode<ISA>>>) {}

    node_kind get_type() const { return type; }

    LoopTreeFunction get_fn() const
    {
        return [this](std::map<std::string, float*> tensors) {
            auto input  = this->input;
            auto output = this->output;
            LN_LOG(DEBUG) << "Hit transpose\n";
            float* A = tensors.at(input);
            float* C = tensors.at(output);
            // TODO(j): generalize to other ops supported....
            LN_LOG(DEBUG) << "(C:" << C[0] << ") * (A:" << A[0] << ")"
                          << "\n";
            C[0] = A[0];
        };
    }
};


template <class ISA>
std::shared_ptr<ProgramNode<ISA>>
merge_loop_into_jitter(for_loop_node<ISA>* node, ComputeNode<ISA>* child,
                       std::map<std::string, int>                   sizes,
                       std::map<std::string, std::set<std::string>> formulas)
{
    auto ret = new jitted_loop_nest_node<ISA>(node, child, sizes, formulas);
    return ret;
    // delete ret;
}

template <class ISA>
std::shared_ptr<ProgramNode<ISA>>
merge_loop_into_jitter(for_loop_node<ISA>*         node,
                       jitted_loop_nest_node<ISA>* child)
{
    return new jitted_loop_nest_node<ISA>(node, child);
}

template <class ISA>
void compile_loop_nests(std::shared_ptr<ProgramNode<ISA>> node)
{
    switch (node->get_type())
    {
    case node_kind::compute_jitter_type:
        dynamic_cast<jitted_loop_nest_node<ISA>*>(node)->jit_compile();
        break;
    case node_kind::transpose_jitter_type:
        // TODO(j): handle
        // static_cast<TransposeJitterNode *>(node)->jit_compile();
        break;
    case node_kind::for_type:
        for (auto child : node->get_children())
        {
            compile_loop_nests(child);
        }
        break;
    default:
        // do nothing
        break;
    }
}
