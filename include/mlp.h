#ifndef MLP_H
#define MLP_H

#include <iostream>
#include <string>
#include <armadillo>

using namespace std;
using namespace arma;


/**  \class mpl mlp.h mlp.cpp
 *
 * Multilayer perceptron with one hidden layer
 *
 */
class mlp
{
    public:
        mlp(string settings_file); //!< The constructor
        ~mlp();  //!< The destructor
        mlp(const mlp& other); //!< The copy constructor
        mlp& operator=(const mlp& other); //!< The assignment operator

        unsigned int nInputs; //!< Number of inputs to MLP plus bias
        unsigned int nHLneurons; //!< Number of neurons in hidden layer in MPL
        unsigned int nWeights; //!< Number of weights in MPL
        unsigned int nTargets; //!< Number of targets

        unsigned int Pr_Number; //!< Problem selection for calculating the fitness function

        colvec weights; //!< Weights in MPL, size = (nInputs * nHLneurons) + (nHLneurons + 1) = (weights from IN to hidden) + (weights from hidden and bias to OUT)
        colvec activations; //!< Activations, size = nHLneurons + 1, activations(0) is for the first hidden layer neuron, activations(nHLneurons) is for output MLP neuron
        colvec neuronOutputs; //!< Neurons outputs using transformation with activation function, size = nHLneurons

        mat Inputs; //!< Targed input data to MLP, size: ncol = nInputs, nrow = nTargets, the last column has to be filled with 1 (i.e. by bias)
        colvec obsOutputs; //!< Observed outputs from MLP, size = nTargets
        colvec simOutputs; //!< Simulated outputs from MLP, size = nTargets

        mat trans_Inputs; //!< Transformed input data to MLP, size: ncol = nInputs, nrow = nTargets, the last column has to be filled with 1 (i.e. by bias)

        string directoryIn; //!< Base directory for input data
        string directoryOut; //!< Base directory for results output

        void create_input(string load_file,string file_name,string save_file); //!< Create input data to the final MLP from outputs form each MLP

        void read_file(string file_name); //!< Read measured input and output data from file
        void read_weights(string file_name,unsigned int number_model); //!< Read weights from file

        void orig_to_trans(); //!!< Transformation of the original data
        double trans_to_orig(double trans_data); //!!< Transformation back to the original data

        double calc_output(colvec help_weights, rowvec help_inputs); //!< Method for calculation of the activation and final transformed output from the MLP
        void calc_output_HL(colvec help_activ); //!< Method for calculation of the neuron output after activation

        double calc_fittnes(unsigned int crit,colvec var_obs, colvec var_mod); //!< Compute fitness
        void out_fittness(double value, unsigned int first_col, unsigned int second_col, string save_file); //!< Writing the fitness to the file
        void out_outputs(mat data, string save_file); //!< Writing the simulation outputs to the file

        void runMLP(); //!< Run MLP model

    protected:
    private:
};

#endif // MLP_H
