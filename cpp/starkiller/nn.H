#ifndef NET_TRAINING_H_
#define NET_TRAINING_H_

#include <starkiller.H>

#include <torch/torch.h>


struct HiddenNetImpl : public torch::nn::Module
{
    HiddenNetImpl(int64_t n_independent, int64_t n_dependent,
                  int64_t n_hidden, int64_t hidden_depth) {

        // Construct and register the layers
        input_layer = register_module("input", torch::nn::Linear(n_independent, n_hidden));
        for (int i = 0; i < hidden_depth; ++i) {
            hidden_layer->push_back(torch::nn::Linear(n_hidden,n_hidden));
        }
        hidden_layer = register_module("hidden", hidden_layer);
        output_layer = register_module("output", torch::nn::Linear(n_hidden, n_dependent));
    };

    // Implement the forward method.
    torch::Tensor forward(torch::Tensor x) {
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

class NetTraining {

public:

  // constructor
  NetTraining();

  // destructor
  virtual ~NetTraining();

  // initialize data
  void init();


  /* private variables */

  ReactionSystem system;
  ReactionSystem system_test;
};

#endif
