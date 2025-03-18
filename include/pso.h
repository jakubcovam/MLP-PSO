#ifndef PSO_H
#define PSO_H

#include <iostream>
#include <string>
#include <armadillo>

using namespace std;
using namespace arma;


/** \class pso pso.cpp
 *
 * PSO settings and variables
 *
 */
class pso
{
    public:

        enum { ConstrFactor,ConstantW,RandomW,LinTimeVaryingW,ChaoticW,ChaoticRandW,NonlinTimeConstW,\
               NonlinTimeVaryingW,AdaptW,AdaptParticleW } SCPSO_type;
        enum { INV_emp_dist} quantile_type;
        enum { MSE, MAE, MAPE, NS, LNNS, MRE, tPI, PI, PID, dRMSE, PID1 } crit_type_PSO;

        pso( string settings_file,mlp& ann);
        ~pso();
        pso(const pso& other);
        pso& operator=(const pso& other);

        unsigned int N_Complexes; //!< Number of complexes
        unsigned int Dim; //!< Problem dimension -- number of model parameters

        double lo_limit; //!< lower limit for initialization
        double up_limit; //!< upper limit for initialization;

        double K; //!< K parameter of PSO (constriction factor)
        double C; //!< C parameter of PSO (acceleration constant) for constriction factor
        unsigned int Cw; //!< C parameter of PSO (acceleration constant) for inertia weight
        double W; //!< w parameter of PSO (inertia weight)

        unsigned int all_n_Population; //!< The total sum of all Populations, size = (nPop) * N_Complexes
        unsigned int n_one_population; //!< Total number of one population members
        unsigned int all_n_Param; //!< The total number of all parameters in Population, size = Dim * all_n_Population

        uvec nPop; //!< Vector of population numbers, its size controlled by the number complexes (case unequal complexes)

        mat Model_param_Parents_PSO; //!< Matrix of all information about the population (position, velocity, pbest)
        mat pbest; //!< The best position found by the ith particle

        colvec best_model; //!< Vector of the best model
        colvec best_model_ens; //!< The best model for each ensemble
        double best_model_ens_final; //!< The final best model out of all ensembles
        colvec best_simOutputs; //!< The best simulated outputs from each ensemble
        mat best_simOutputs_fin; //!< The best simulated outputs from each ensemble saved in a final matrix for writing into the file
        colvec best_weights; //!< The best weights achieved at each ensemble
        mat best_weights_fin; //!< The best weights from each ensemble saved in a final matrix for writing into the file
        colvec best_best_simOutputs_fin; //!< The best simulated outputs from all ensembles based on OF
        colvec best_best_weights_fin; //!< The best weights from all ensembles based on OF

        colvec fitness; //!< Stored fitness of Parents models (1all_n_Population), LAST ROW FITTNES
        unsigned int number_func_evaluation; //!< Number of function evaluations
        unsigned int Number_of_generations_in_one_complex; //!< Maximum number of allowed function evaluations in each partial complex population
        unsigned int max_number_of_shuffles; //!< Maximum number of shuffling
        unsigned int max_function_eval; //!< Maximum number of function evaluation

        cube Model_Comp_PSO; //!< Population of particles, where at each slide is one complex

        unsigned int PRoblem_Number; //!< Problem selection for calculating the fitness function
        double searched_minimum; //!< Forgiven problem searched value of global minimum

        unsigned int ensemble; //!< Number of simulation runs
        unsigned int SCPSO_selection; //!< Chosen variant of PSO

        string directory; //!< Base directory for results output
        string path_out_file; //!< Directory for ensemble results outcome
        string SCPSO_NAME; //!< Indetification of SCPSO type
        stringstream EN_INFO; //!< Used parameters

        mat PSO_ConstrFactor(mat POpulation,mlp& ann); //!< PSO/constriction factor
        mat PSO_ConstantW(mat POpulation,mlp& ann); //!< PSO/constant inertia weight
        mat PSO_RandomW(mat POpulation,mlp& ann); //!< PSO/random inertia weight
        mat PSO_LinTimeVaryingW(mat POpulation,mlp& ann); //!< PSO/linear time varying inertia weight
        mat PSO_ChaoticW(mat POpulation,mlp& ann); //!< PSO/linear time varying inertia weight with random changes
        mat PSO_ChaoticRandW(mat POpulation,mlp& ann); //!< PSO/linear time varying inertia weight with random changes
        mat PSO_NonlinTimeConstW(mat POpulation,mlp& ann); //!< PSO/nonlinear time varying inertia weight
        mat PSO_NonlinTimeVaryingW(mat POpulation,mlp& ann); //!< PSO/nonlinear time varying inertia weight
        mat PSO_AdaptW(mat POpulation,mlp& ann); //!< PSO/adaptive inertia weight
        mat PSO_AdaptParticleW(mat POpulation,mlp& ann); //!< PSO/adapt inertia weight for each particle

        void SC_PSO_ConstrFactor(unsigned int ensemble_number,mlp& ann); //!< SCE PSO/constriction factor
        void SC_PSO_ConstantW(unsigned int ensemble_number,mlp& ann); //!< SCE PSO/constant inertia weight
        void SC_PSO_RandomW(unsigned int ensemble_number,mlp& ann); //!< SCE PSO/random inertia weight
        void SC_PSO_LinTimeVaryingW(unsigned int ensemble_number,mlp& ann); //!< SCE PSO/linear time varying inertia weight
        void SC_PSO_ChaoticW(unsigned int ensemble_number,mlp& ann); //!< SCE PSO/linear time varying inertia weight with random changes
        void SC_PSO_ChaoticRandW(unsigned int ensemble_number,mlp& ann); //!< SCE PSO/linear time varying inertia weight with random changes
        void SC_PSO_NonlinTimeConstW(unsigned int ensemble_number,mlp& ann); //!< SCE PSO/nonlinear time varying inertia weight
        void SC_PSO_NonlinTimeVaryingW(unsigned int ensemble_number,mlp& ann); //!< SCE PSO/nonlinear time varying inertia weight
        void SC_PSO_AdaptW(unsigned int ensemble_number,mlp& ann); //!< SCE PSO/adaptive inertia weight
        void SC_PSO_AdaptParticleW(unsigned int ensemble_number,mlp& ann); //!< SCE PSO/adaptive inertia weight for each particle

        void initialize_all_POpulation(); //!< Inicilaization of all members of Population. Initialize the position, velocity, and pbest
        void make_mat_from_Compl(); //!< Combining the complexes to the parents matrix
        void sort_model_param_parents_mat(); //!< Sorting the parent matrix according to the fittness function
        void make_Compl_from_mat(); //!< Creating the complexes from matix (P1,P2,P3,...,PN,P1,P2,P3,...,Pn,...)
        double calc_fit(colvec help_col,mlp& ann); //!< Calculate fitness value called from PSO method

        unsigned int rand_int(unsigned int n); //!< Generates integer random number
        double randu_interval(); //!< Random number generator (0,1)
        template<class my_vec> my_vec random_perm(unsigned int n); //!< Random permutations
        //uvec random_perm(unsigned int n)

        void run_ensemble(mlp& ann); //!< Compute ensemble run

        unsigned int quantile_selection; //!< Type of quantile estimates
        template<class my_vec>  my_vec quantiles_(my_vec data, my_vec probs); //!< Template for quantile estimation
        template<class my_vec> my_vec unique_(my_vec data); //!< Method for quantile estimation of 7 type Hydrman et al see cal_quantile description

        void calc_quantiles(); //!< Calculation the quantiles and printing them to the file
        void calc_quantiles_fast_small();

    protected:
    private:
};

#endif // PSO_H
