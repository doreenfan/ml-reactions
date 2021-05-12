#ifndef NET_TRAINING_H_
#define NET_TRAINING_H_

#include <starkiller.H>

#include <torch/torch.h>


struct HiddenNetImpl : public torch::nn::Module
{
    HiddenNetImpl(int64_t n_independent, int64_t n_dependent,
                  int64_t n_hidden, int64_t hidden_depth)
    {

        // Construct and register the layers
        input_layer = register_module("input", torch::nn::Linear(n_independent, n_hidden));
        for (int i = 0; i < hidden_depth; ++i) {
            hidden_layer->push_back(torch::nn::Linear(n_hidden,n_hidden));
        }
        hidden_layer = register_module("hidden", hidden_layer);
        output_layer = register_module("output", torch::nn::Linear(n_hidden, n_dependent));
    };

    // Implement the forward method.
    torch::Tensor forward(torch::Tensor x)
    {
        // Note: all activation layers are tanh
        // will FIX to allow user defined activations
        x = torch::tanh(input_layer->forward(x));
        for (auto& layer : *hidden_layer) {
            x = torch::tanh(layer->as<torch::nn::Linear>()->forward(x));
        }
        x = torch::tanh(output_layer->forward(x));

        return x;
    };

    // Use one of many "standard library" modules.
    torch::nn::Linear input_layer{nullptr};
    torch::nn::Linear output_layer{nullptr};
    torch::nn::ModuleList hidden_layer;

};

TORCH_MODULE(HiddenNet);


// NetTraining class

class NetTraining
{
  private:

    torch::Tensor x, y, dydx;
    torch::Tensor x_test, y_test;

    int64_t n_independent, n_dependent;
    int64_t n_hidden, hidden_depth;
    std::string optimizer_type;

  public:

    // constructor
    explicit NetTraining(int size, int size_test)
    {
        x = torch::empty({size,1}, torch::TensorOptions().requires_grad(true));
        y = torch::empty({size,NumSpec+2}, torch::TensorOptions().requires_grad(true));
        dydx = torch::empty({size,NumSpec+2});

        x_test = torch::empty({size_test,1});
        y_test = torch::empty({size_test,NumSpec+2});
    };

    // destructor
    //~NetTraining();

    // initialize parameters
    void init(int64_t n_i, int64_t n_d,
              int64_t n_h, int64_t h_depth,
              std::string op_type);

    // initialize data
    void init_data(const Vector<MultiFab>& input, const Vector<MultiFab>& output,
                   const Vector<MultiFab>& outputdot);

    // initialize test data
    void init_test(const Vector<MultiFab>& input_test,
                   const Vector<MultiFab>& output_test);

    // initialize neural net
    // void init_net();

    // training loop
    void train(int NumEpochs, int start_epoch=0);

    // compute gradients for each component
    torch::Tensor get_component_gradient(int n);

    // define custom loss functions
    float mse_loss(torch::Tensor input, torch::Tensor target);
    float rms_weighted_error(torch::Tensor input, torch::Tensor target,
                             torch::Tensor solution);


};

#endif
