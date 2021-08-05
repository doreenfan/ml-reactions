#include <nn.H>

// initialize parameters
void NetTraining::init(int64_t n_i, int64_t n_d,
                       int64_t n_h, int64_t h_depth, std::string op_type)
{
    n_independent = n_i;
    n_dependent = n_d;
    n_hidden = n_h;
    hidden_depth = h_depth;

    optimizer_type = op_type;
}

// initialize data
void NetTraining::init_data(const Vector<MultiFab>& input, const Vector<MultiFab>& output,
                            const Vector<MultiFab>& outputdot)
{
    // convert input, output, and outputdot to Tensors

}

// initialize test data
void NetTraining::init_test(const Vector<MultiFab>& input_test,
                            const Vector<MultiFab>& output_test)
{

}

// training loop
void NetTraining::train(int NumEpochs, int start_epoch)
{

}

#if 0
// compute gradients for each component
torch::Tensor NetTraining::get_component_gradient(int n)
{
    torch::Tensor dpndx;

    return (dpndx);
}

// define custom loss functions
float NetTraining::mse_loss(torch::Tensor input, torch::Tensor target)
{
    return (0.0);
}

float NetTraining::rms_weighted_error(torch::Tensor input, torch::Tensor target,
                                       torch::Tensor solution)
{
    return (0.0);
}
#endif
