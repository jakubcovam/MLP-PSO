#include "mlp.h"
#include "pso.h"


/**
 * The constructor
 * \param settings_file: file with PSO settings
 */
pso::pso(string settings_file,mlp& ann)
{
    string trash;

    ifstream pso_inform(settings_file.c_str());
    if(!pso_inform) { // osetreni jestli je soubor mozne otevrit
      cout << "\nUnable to open file :"<< settings_file.c_str() <<" with initial settings for computations.";
      exit(EXIT_FAILURE);
    }
    pso_inform >> scientific;
    pso_inform >> trash >> trash >> N_Complexes >> trash >> n_one_population;
    //pso_inform >> trash >> PRoblem_Number;
    //if((PRoblem_Number >=0)&&(PRoblem_Number <=5)) Nfunc = 1;
    pso_inform >> trash >> Number_of_generations_in_one_complex;
    pso_inform >> trash >> max_number_of_shuffles;
    pso_inform >> trash >> K >> C >> Cw;
    pso_inform >> trash >> ensemble;
    pso_inform >> trash >> SCPSO_selection;
    pso_inform >> trash >> trash;
    pso_inform >> trash >> directory;
//    pso_inform >> trash >> path_out_file;

    pso_inform.close();

    PRoblem_Number = ann.Pr_Number;
    Dim = size(ann.weights,0);
    //cout << "Dim = " << Dim << endl;
    all_n_Population = N_Complexes * n_one_population;
    all_n_Param = Dim *all_n_Population;

   /* cout << endl << "**** REVIEW of PSO settings ****"<< endl;
    cout << scientific;
    cout << "\nPSO initialization with following settings from file: " << settings_file.c_str();
    cout << "\nNumber of complexes: " << N_Complexes << "\nNumber of members in one Population: " << n_one_population << endl;
    cout << "The total population: " << N_Complexes * n_one_population << "\nDimension of the problem: " << Dim  <<endl;
    cout << "Total number of real parameters in population: " << all_n_Param << "\nLower bound for parameter values: " << lo_limit << endl;
    cout << "Upper bound for parameter limit: " << up_limit<< endl;
    cout << "Opt_crit_0-MSE_1-MAE_2-MAPE_3-NS_4-LNNS_5-MRE: " << PRoblem_Number << endl;
    cout << "Funtion evaluations in one complex is " <<    Number_of_generations_in_one_complex << endl;
    cout << "Maximum allowed shuffling  is " <<    max_number_of_shuffles << endl;
    cout << "Read parameter values K - C - Cw: " << K << " "<< C  << " "<< Cw << endl;
    cout << "The ensemble " << ensemble << endl;
    cout << "Algorithm_selection: " << SCPSO_selection << endl;
    cout << "PSO_selection_0-ConstrFactor_1-ConstantW_2-RandomW_3-LinTimeVaryingW_4-ChaoticW_ " << endl;
    cout << "PSO_selection_5-ChaoticRandW_6-NonlinTimeConstW_7-NonlinTimeVaryingW_8-AdaptW_9-APartW " << endl;
    cout << "Directory_for_output: " << directory.c_str() << endl;
   */

    lo_limit = -1;
    up_limit = 1;

    Model_Comp_PSO.set_size(3*Dim+2,n_one_population,N_Complexes);
    Model_Comp_PSO.fill(99999999);

    Model_param_Parents_PSO.set_size(3*Dim+2, all_n_Population);
    Model_param_Parents_PSO.fill(99999999);

    pbest.set_size(Dim+1,n_one_population);
    //pbest.fill(10e8);

    best_model.set_size(3*Dim+2);
    best_model_ens.set_size(3*Dim+2);
    best_model_ens_final = 99999999;
    best_simOutputs.set_size(ann.nTargets);
    best_simOutputs_fin.set_size(ann.nTargets,ensemble);
    best_weights.set_size(ann.nWeights);
    best_weights_fin.set_size(ann.nWeights,ensemble);
    best_best_simOutputs_fin.set_size(ann.nTargets);
    best_best_weights_fin.set_size(ann.nWeights);

    number_func_evaluation = 0;
    path_out_file += "INITIAL";

    switch (SCPSO_selection)
    {
        case ConstrFactor:
            SCPSO_NAME = "_PSO_ConstrFactor_";
            EN_INFO  << "_DIM-" << Dim << "_f" << PRoblem_Number+1 <<"_N-COMP-" << N_Complexes << "_SHUF-GEN-" << max_number_of_shuffles;
            EN_INFO << "_ONE-GEN-" << Number_of_generations_in_one_complex << "_POP1COM-" << n_one_population << "_K-" << K << "_C-" << C;
            break;
        case ConstantW:
            SCPSO_NAME = "_PSO_ConstantW_";
            EN_INFO  << "_DIM-" << Dim << "_f" << PRoblem_Number+1 <<"_N-COMP-" << N_Complexes << "_SHUF-GEN-" << max_number_of_shuffles;
            EN_INFO << "_ONE-GEN-" << Number_of_generations_in_one_complex << "_POP1COM-" << n_one_population << "_Cw-" << Cw;
            break;
        case RandomW:
            SCPSO_NAME = "_PSO_RandomW_";
            EN_INFO  << "_DIM-" << Dim << "_f" << PRoblem_Number+1 <<"_N-COMP-" << N_Complexes << "_SHUF-GEN-" << max_number_of_shuffles;
            EN_INFO << "_ONE-GEN-" << Number_of_generations_in_one_complex << "_POP1COM-" << n_one_population << "_Cw-" << Cw ;
            break;
        case LinTimeVaryingW:
            SCPSO_NAME = "_PSO_LinTimeVaryingW_";
            EN_INFO  << "_DIM-" << Dim << "_f" << PRoblem_Number+1 <<"_N-COMP-" << N_Complexes << "_SHUF-GEN-" << max_number_of_shuffles;
            EN_INFO << "_ONE-GEN-" << Number_of_generations_in_one_complex << "_POP1COM-" << n_one_population << "_Cw-" << Cw ;
            break;
        case ChaoticW:
            SCPSO_NAME = "_PSO_ChaoticW_";
            EN_INFO  << "_DIM-" << Dim << "_f" << PRoblem_Number+1 <<"_N-COMP-" << N_Complexes << "_SHUF-GEN-" << max_number_of_shuffles;
            EN_INFO << "_ONE-GEN-" << Number_of_generations_in_one_complex << "_POP1COM-" << n_one_population << "_Cw-" << Cw ;
            break;
        case ChaoticRandW:
            SCPSO_NAME = "_PSO_ChaoticRandW_";
            EN_INFO  << "_DIM-" << Dim << "_f" << PRoblem_Number+1 <<"_N-COMP-" << N_Complexes << "_SHUF-GEN-" << max_number_of_shuffles;
            EN_INFO << "_ONE-GEN-" << Number_of_generations_in_one_complex << "_POP1COM-" << n_one_population << "_Cw-" << Cw ;
            break;
        case NonlinTimeConstW:
            SCPSO_NAME = "_PSO_NonlinTimeConstW_";
            EN_INFO  << "_DIM-" << Dim << "_f" << PRoblem_Number+1 <<"_N-COMP-" << N_Complexes << "_SHUF-GEN-" << max_number_of_shuffles;
            EN_INFO << "_ONE-GEN-" << Number_of_generations_in_one_complex << "_POP1COM-" << n_one_population << "_Cw-" << Cw ;
            break;
        case NonlinTimeVaryingW:
            SCPSO_NAME = "_PSO_NonlinTimeVaryingW_";
            EN_INFO  << "_DIM-" << Dim << "_f" << PRoblem_Number+1 <<"_N-COMP-" << N_Complexes << "_SHUF-GEN-" << max_number_of_shuffles;
            EN_INFO << "_ONE-GEN-" << Number_of_generations_in_one_complex << "_POP1COM-" << n_one_population << "_Cw-" << Cw ;
            break;
        case AdaptW:
            SCPSO_NAME = "_PSO_AdaptW_";
            EN_INFO  << "_DIM-" << Dim << "_f" << PRoblem_Number+1 <<"_N-COMP-" << N_Complexes << "_SHUF-GEN-" << max_number_of_shuffles;
            EN_INFO << "_ONE-GEN-" << Number_of_generations_in_one_complex << "_POP1COM-" << n_one_population << "_Cw-" << Cw ;
            break;
        case AdaptParticleW:
            SCPSO_NAME = "_PSO_AdaptParticleW_";
            EN_INFO  << "_DIM-" << Dim << "_f" << PRoblem_Number+1 <<"_N-COMP-" << N_Complexes << "_SHUF-GEN-" << max_number_of_shuffles;
            EN_INFO << "_ONE-GEN-" << Number_of_generations_in_one_complex << "_POP1COM-" << n_one_population << "_Cw-" << Cw  ;
            break;
    }

    switch (PRoblem_Number)
    {
        case 0:  //MSE
            searched_minimum = 0.0;
            break;
        case 1:  //MAE
            searched_minimum = 0.0;
            break;
        case 2:  //MAPE
            searched_minimum = 0.0;
            break;
        case 3:  //NS
            searched_minimum = 0.0; //1.0
            break;
        case 4:  //LNNS
            searched_minimum = 0.0; //1.0
            break;
        case 5:  //MRE
            searched_minimum = 0.0;
            break;
        case 6:  //tPI
            searched_minimum = 0.0;
            break;
        case 7:  //PI
            searched_minimum = 0.0; //1.0
            break;
        case 8:  //PID
            searched_minimum = 0.0;
            break;
        case 9:  //dRMSE
            searched_minimum = 0.0;
            break;
        case 10:  //PID1
            searched_minimum = 0.0;
            break;
        default:
            break;
    }
}

/**
 * The destructor
 */
pso::~pso()
{
    //dtor
}

/**
 * The copy constructor
 */
pso::pso(const pso& other)
{
    //copy ctor
        N_Complexes = other.N_Complexes;
        Dim = other.Dim;

        lo_limit = other.lo_limit;
        up_limit = other.up_limit;

        K = other.K;
        C = other.C;
        Cw = other.Cw;

        all_n_Population = other.all_n_Population;
        n_one_population = other.n_one_population;
        all_n_Param = other.all_n_Param;

        nPop = other.nPop;

        Model_param_Parents_PSO = other.Model_param_Parents_PSO;
        pbest = other.pbest;

        best_model = other.best_model;
        best_model_ens = other.best_model_ens;
        best_model_ens_final = other.best_model_ens_final;
        best_simOutputs = other.best_simOutputs;
        best_simOutputs_fin = other.best_simOutputs_fin;
        best_weights = other.best_weights;
        best_weights_fin = other.best_weights_fin;
        best_best_simOutputs_fin = other.best_best_simOutputs_fin;
        best_best_weights_fin = other.best_best_weights_fin;

        fitness = other.fitness;
        number_func_evaluation = other.number_func_evaluation;
        Number_of_generations_in_one_complex = other.Number_of_generations_in_one_complex;
        max_number_of_shuffles = other.max_number_of_shuffles;

        Model_Comp_PSO = other.Model_Comp_PSO;

        PRoblem_Number = other.PRoblem_Number;
        //Nfunc = other.Nfunc;

        ensemble = other.ensemble;
        SCPSO_selection = other.SCPSO_selection;

        directory = other.directory;
        path_out_file = other.path_out_file;

        quantile_selection = other.quantile_selection;

        searched_minimum = other.searched_minimum;
}

/**
 * The assignment operator
 */
pso& pso::operator=(const pso& other)
{
    if (this == &other) {
            return *this; // handle self assignment
    }  else {
        N_Complexes = other.N_Complexes;
        Dim = other.Dim;

        lo_limit = other.lo_limit;
        up_limit = other.up_limit;

        K = other.K;
        C = other.C;
        Cw = other.Cw;

        all_n_Population = other.all_n_Population;
        n_one_population = other.n_one_population;
        all_n_Param = other.all_n_Param;

        nPop = other.nPop;

        Model_param_Parents_PSO = other.Model_param_Parents_PSO;
        pbest = other.pbest;

        best_model = other.best_model;
        best_model_ens = other.best_model_ens;
        best_model_ens_final = other.best_model_ens_final;
        best_simOutputs = other.best_simOutputs;
        best_simOutputs_fin = other.best_simOutputs_fin;
        best_weights = other.best_weights;
        best_weights_fin = other.best_weights_fin;
        best_best_simOutputs_fin = other.best_best_simOutputs_fin;
        best_best_weights_fin = other.best_best_weights_fin;

        fitness = other.fitness;
        number_func_evaluation = other.number_func_evaluation;
        Number_of_generations_in_one_complex = other.Number_of_generations_in_one_complex;
        max_number_of_shuffles = other.max_number_of_shuffles;

        Model_Comp_PSO = other.Model_Comp_PSO;

        PRoblem_Number = other.PRoblem_Number;
        //Nfunc = other.Nfunc;

        ensemble = other.ensemble;
        SCPSO_selection = other.SCPSO_selection;

        directory = other.directory;
        path_out_file = other.path_out_file;

        quantile_selection = other.quantile_selection;
        searched_minimum = other.searched_minimum;
    }
    return *this;
}

/**
 * The initialization of all random Population
 */
void pso::initialize_all_POpulation()
{
  colvec my_init = random_perm<colvec>(all_n_Param);
  colvec help_vec;

  help_vec.set_size(all_n_Param);
  help_vec.randu();

  my_init -= help_vec;
  my_init = (up_limit - lo_limit) * my_init / all_n_Param + lo_limit;

  //!! Initializing the Model param Parents PSO matrix - position
  unsigned int help_var =0;
  for (unsigned int i =0; i < all_n_Population ;i++ ){
    for (unsigned int j =0; j<Dim ;j++ ){
      Model_param_Parents_PSO(j,i) = my_init(help_var);
      help_var++;
    }
  }
  colvec help_col;
  help_col.set_size(Dim);
  for(unsigned int i=0; i< all_n_Population;i++){
    help_col = Model_param_Parents_PSO.submat(0,i,Dim-1,i);
    //cout <<endl <<help_col << endl;
    //mj Model_param_Parents_PSO(Dim,i) = fittnes(help_col);
    Model_param_Parents_PSO(Dim,i) = 10e5;
  }

  //!! Initializing the Model param Parents PSO matrix - velocity
  for (unsigned int i=0; i<all_n_Population ;i++) {
    for (unsigned int j=(Dim+1); j<(2*Dim+1); j++) {
      //Model_param_Parents_PSO(j,i) = lo_limit + (up_limit - lo_limit)*randu_interval();
      Model_param_Parents_PSO(j,i) = up_limit;
    }
  }
//cout <<endl << Model_param_Parents_PSO << endl;

  //!! Initializing the Model param Parents PSO matrix - pbest
  for (unsigned int i=0; i<all_n_Population ;i++) {
    for (unsigned int j=(2*Dim+1); j<(3*Dim+1); j++) {
      Model_param_Parents_PSO(j,i) = 10e5;
    }
  }
  colvec help_pbest;
  help_pbest.set_size(Dim);
  for(unsigned int i=0; i< all_n_Population;i++){
    help_pbest = Model_param_Parents_PSO.submat(2*Dim+1,i,3*Dim,i);
    //mj Model_param_Parents_PSO(3*Dim+1,i) = fittnes_pbest(help_pbest);
    Model_param_Parents_PSO(3*Dim+1,i) = 10e5;
  }

  make_Compl_from_mat();
}

/**
 * Random permutation according to Richard Durstenfeld in 1964 in Communications of the ACM volume 7, issue 7, as "Algorithm 235: Random permutation"
 * \param n: number of data
 */
  template <class my_vec> my_vec pso::random_perm(unsigned int n)
{
  unsigned int j=0;
  my_vec shuffled_index, tmp;

  shuffled_index.set_size(n);
  for (unsigned int i=0;i<n ;i++ ){
    shuffled_index(i) = i+1;
    }

   tmp.set_size(1);
   tmp.fill(1);

  for (unsigned int i = n - 1; i > 0; i--) {
    j = rand_int(i);
    tmp(0) = shuffled_index(j);
    shuffled_index(j) = shuffled_index(i);
    shuffled_index(i) = tmp(0);
  }

  return(shuffled_index);
}

/**
 * Random integer number generator
 */
unsigned int pso::rand_int(unsigned int n)
{
  unsigned int limit = RAND_MAX - RAND_MAX % n;
  unsigned int rnd;

//  srand((unsigned)time(0));
  //cout << "\n time "<< time(0)<< endl;

  do {
    rnd = rand();
  } while (rnd >= limit);

  return rnd % n;
}

/**
 * After sorting the population according to its fittness -- shuffling the population from Parent matrix to complexes
 */
void pso::make_Compl_from_mat()
{
   unsigned int help_var=0;

   for (unsigned int j=0; j< n_one_population ; j++){
      for (unsigned int i =0; i< N_Complexes ;i++ ){
         Model_Comp_PSO.subcube(0,j,i,(3*Dim+1),j,i) = Model_param_Parents_PSO.col(help_var);
         help_var++;
      }
   }
}

/**
 * Creating the Parent matrix by combining the Complexes of small Populations
 */
void pso::make_mat_from_Compl()
{
    unsigned int left_col=0, right_col;
    right_col = n_one_population -1;

    for (unsigned int j=0; j<N_Complexes ;j++ ){
        Model_param_Parents_PSO.submat(0,left_col,(3*Dim+1),right_col) = Model_Comp_PSO.slice(j);
        left_col = right_col+1;
        right_col += n_one_population;
    }
      //cout << endl <<"make_mat_from_compl\n" << Model_param_Parents_PSO<< endl ;
}

/**
 * Sorting columns of Parent matrix according to fittnes
 */
void pso::sort_model_param_parents_mat()
{
      rowvec fit_ness;

      fit_ness = Model_param_Parents_PSO.row(Dim);
      //fit_ness = abs(Model_param_Parents_PSO.row(Dim)-searched_minimum);
//    cout << fit_ness;
      umat indexes = sort_index(fit_ness);
//    cout <<  "\n sorted indexes\n"  << indexes;
//      cout << "model_param: PRED: \n"<< Model_param_Parents_PSO << endl;

      mat Help_Model_param_parents_PSO;
      Help_Model_param_parents_PSO.set_size(3*Dim+2, all_n_Population);

      for (unsigned int i = 0; i< all_n_Population ;i++ ){
         Help_Model_param_parents_PSO.col(i) = Model_param_Parents_PSO.col(indexes(i));
      }

      Model_param_Parents_PSO = Help_Model_param_parents_PSO;
//      cout << "model_param: PO: \n"<< Model_param_Parents_PSO << endl;
      best_model = Model_param_Parents_PSO.col(0);

}

/**
 * Random number generator provides double value in interval (0.1)
 */
double pso::randu_interval()
{
    double value = 0.0;
    unsigned int LIM_MAX;

    LIM_MAX =   1+ (static_cast<unsigned int>( RAND_MAX ) );
    value = static_cast<double>(rand() ) / (static_cast<double>(LIM_MAX));

    return (value);
}

/**
 * Calculate fitness value called from PSO method
 * \param help_col: colvec with weights, which are optimized
 * \return value: calculated fitness value
 */
double pso::calc_fit(colvec help_col, mlp& ann)
{
    double value = 0.0;

    colvec help_simOutputs;
    help_simOutputs.set_size(ann.nTargets);
    help_simOutputs.fill(1000);

    for (unsigned int k=0; k<ann.nTargets; k++) {
      //ann.calc_activation(help_col,ann.trans_Inputs.row(k));
      help_simOutputs(k) = ann.calc_output(help_col,ann.trans_Inputs.row(k));
      ann.simOutputs(k)= ann.trans_to_orig(as_scalar(help_simOutputs(k)));
    }
    value = ann.calc_fittnes(PRoblem_Number,ann.obsOutputs,ann.simOutputs);

    return (value);

}


/**
 * PSO VERSIONS
 */

/**
 * PSO/constriction factor
 */
mat pso::PSO_ConstrFactor(mat POpulation,mlp& ann)
{
    mat offsprings;
    mat velocity;

    offsprings.set_size(Dim+1,n_one_population);
    velocity.set_size(Dim,n_one_population);

    offsprings.fill(44);
    velocity.fill(up_limit);

    for (unsigned int j=0;j<Number_of_generations_in_one_complex;j++ ){
            for (unsigned int i=0; i<n_one_population ; i++){  //n_one_population = Number_of_Members_in_one_Complex
                if (abs(as_scalar(POpulation(Dim,i))-searched_minimum) < abs(as_scalar(POpulation(3*Dim+1,i))-searched_minimum)) {
                    pbest.col(i) = POpulation.submat(0,i,Dim,i);
                }
                else {
                    pbest.col(i) = POpulation.submat(2*Dim+1,i,3*Dim+1,i);
                }

                for (unsigned int s=0; s<Dim; s++ ){
                    velocity(s,i) = K * POpulation((s+Dim+1),i) + C * randu_interval()*(pbest(s,i) - POpulation(s,i)) + C * randu_interval() * (best_model(s) - POpulation(s,i));
                    velocity(s,i) = min(max(velocity(s,i),lo_limit),up_limit);
                    offsprings(s,i) = POpulation(s,i) + velocity(s,i);

                    if((as_scalar(offsprings(s,i))< lo_limit) || (as_scalar(offsprings(s,i))>up_limit)) {
                        offsprings(s,i) = POpulation(s,i);
                    }
                 }//end Dim loop

                 colvec help_col;
                 help_col.set_size(Dim);
                 help_col = offsprings.submat(0,i,Dim-1,i);

                 offsprings(Dim,i) = calc_fit(help_col,ann);
                 number_func_evaluation++;

                  if(abs(as_scalar(offsprings(Dim,i))-searched_minimum) < abs(as_scalar(POpulation(Dim,i))-searched_minimum)) {
                    POpulation.submat(0,i,Dim,i) = offsprings.col(i);
                    POpulation.submat(Dim+1,i,2*Dim,i) = velocity.col(i);
                    POpulation.submat(2*Dim+1,i,3*Dim+1,i) = pbest.col(i);
                    if (abs(as_scalar(offsprings(Dim,i))-searched_minimum) <= abs(as_scalar(best_model(Dim))-searched_minimum)) {
                        best_model = offsprings.col(i);
                        best_simOutputs = ann.simOutputs;
                    }
                  }//end of evaluation of fittnes condition
              }//end of offspring population loop
      }//end of max number of function evaluation loop
      //cout << "POpul: \n" << POpulation << endl;
    return POpulation;
}

/**
 * Pure SC PSO/constriction factor
 */
void pso::SC_PSO_ConstrFactor(unsigned int ensemble_number,mlp& ann)
{
    stringstream EN_INFO1;
    EN_INFO1 << "_ens_" << ensemble_number;

    path_out_file +=  "/SC";
    path_out_file += SCPSO_NAME.c_str() +  EN_INFO.str() + EN_INFO1.str();
    initialize_all_POpulation();
//    cout << endl << path_out_file;
    unsigned int kk=0;
    while( (kk < max_number_of_shuffles)){
            sort_model_param_parents_mat();
            make_Compl_from_mat();
            for (unsigned int i =0;i<N_Complexes ; i++){
              Model_Comp_PSO.slice(i) = PSO_ConstrFactor(Model_Comp_PSO.slice(i),ann);
            }
            make_mat_from_Compl();
            kk++;
    }//end while loop
}

/**
 * PSO/constant inertia weight
 */
mat pso::PSO_ConstantW(mat POpulation,mlp& ann)
{
    mat offsprings;
    mat velocity;

    offsprings.set_size(Dim+1,n_one_population);
    velocity.set_size(Dim,n_one_population);

    offsprings.fill(44);
    velocity.fill(up_limit);

    W = 0.7;

    for (unsigned int j=0;j<Number_of_generations_in_one_complex;j++ ){
            for (unsigned int i=0; i<n_one_population ; i++){  //n_one_population = Number_of_Members_in_one_Complex
                if (abs(as_scalar(POpulation(Dim,i))-searched_minimum) < abs(as_scalar(POpulation(3*Dim+1,i))-searched_minimum)) {
                    pbest.col(i) = POpulation.submat(0,i,Dim,i);
                }
                else {
                    pbest.col(i) = POpulation.submat(2*Dim+1,i,3*Dim+1,i);
                }

                for (unsigned int s =0;s <Dim ;s++ ){
                    velocity(s,i) = W * POpulation((s+Dim+1),i) + Cw * randu_interval()*(pbest(s,i) - POpulation(s,i)) + Cw * randu_interval() * (best_model(s) - POpulation(s,i));
                    velocity(s,i) = min(max(velocity(s,i),lo_limit),up_limit);
                    offsprings(s,i) = POpulation(s,i) + velocity(s,i);

                    if((as_scalar(offsprings(s,i))< lo_limit) || (as_scalar(offsprings(s,i))>up_limit)) {
                        offsprings(s,i) = POpulation(s,i);
                    }
                 }//end Dim loop
                 colvec help_col;
                 help_col.set_size(Dim);
                 help_col = offsprings.submat(0,i,Dim-1,i);

                 offsprings(Dim,i) = calc_fit(help_col,ann);
                 number_func_evaluation++;

                  if(abs(as_scalar(offsprings(Dim,i))-searched_minimum) < abs(as_scalar(POpulation(Dim,i))-searched_minimum)) {
                    POpulation.submat(0,i,Dim,i) = offsprings.col(i);
                    POpulation.submat(Dim+1,i,2*Dim,i) = velocity.col(i);
                    POpulation.submat(2*Dim+1,i,3*Dim+1,i) = pbest.col(i);
                    if (abs(as_scalar(offsprings(Dim,i))-searched_minimum) <= abs(as_scalar(best_model(Dim))-searched_minimum)) {
                        best_model = offsprings.col(i);
                        best_simOutputs = ann.simOutputs;

                    }
                  }//end of evaluation of fittnes condition
              }//end of offspring population loop
      }//end of max number of function evaluation loop
      //cout << "POpul: \n" << POpulation << endl;
    return POpulation;
}

/**
 * Pure SC PSO/constant inertia weight
 */
void pso::SC_PSO_ConstantW(unsigned int ensemble_number,mlp& ann)
{
    stringstream EN_INFO1;
    EN_INFO1 << "_ens_" << ensemble_number;

    path_out_file +=  "/SC";
    path_out_file += SCPSO_NAME.c_str() +  EN_INFO.str() + EN_INFO1.str();
    initialize_all_POpulation();
//    cout << endl << path_out_file;
    unsigned int kk=0;
    while( (kk < max_number_of_shuffles)){
            sort_model_param_parents_mat();
            make_Compl_from_mat();
            for (unsigned int i =0;i<N_Complexes ; i++){
              Model_Comp_PSO.slice(i) = PSO_ConstantW(Model_Comp_PSO.slice(i),ann);
            }
            make_mat_from_Compl();
            kk++;
    }//end while loop
}

/**
 * PSO/random inertia weight
 */
mat pso::PSO_RandomW(mat POpulation,mlp& ann)
{
    mat offsprings;
    mat velocity;

    offsprings.set_size(Dim+1,n_one_population);
    velocity.set_size(Dim,n_one_population);

    offsprings.fill(44);
    velocity.fill(up_limit);

    for (unsigned int j=0;j<Number_of_generations_in_one_complex;j++ ){
        W = 0.5 + (randu_interval())/2.0;
            for (unsigned int i=0; i<n_one_population ; i++){  //n_one_population = Number_of_Members_in_one_Complex
                if (abs(as_scalar(POpulation(Dim,i))-searched_minimum) < abs(as_scalar(POpulation(3*Dim+1,i))-searched_minimum)) {
                    pbest.col(i) = POpulation.submat(0,i,Dim,i);
                }
                else {
                    pbest.col(i) = POpulation.submat(2*Dim+1,i,3*Dim+1,i);
                }

                for (unsigned int s =0;s <Dim ;s++ ){
                    velocity(s,i) = W * POpulation((s+Dim+1),i) + Cw * randu_interval()*(pbest(s,i) - POpulation(s,i)) + Cw * randu_interval() * (best_model(s) - POpulation(s,i));
                    velocity(s,i) = min(max(velocity(s,i),lo_limit),up_limit);
                    offsprings(s,i) = POpulation(s,i) + velocity(s,i);

                    if((as_scalar(offsprings(s,i))< lo_limit) || (as_scalar(offsprings(s,i))>up_limit)) {
                        offsprings(s,i) = POpulation(s,i);
                    }
                 }//end Dim loop
                 colvec help_col;
                 help_col.set_size(Dim);
                 help_col = offsprings.submat(0,i,Dim-1,i);

                 offsprings(Dim,i) = calc_fit(help_col,ann);
                 number_func_evaluation++;

                  if(abs(as_scalar(offsprings(Dim,i))-searched_minimum) < abs(as_scalar(POpulation(Dim,i))-searched_minimum)) {
                    POpulation.submat(0,i,Dim,i) = offsprings.col(i);
                    POpulation.submat(Dim+1,i,2*Dim,i) = velocity.col(i);
                    POpulation.submat(2*Dim+1,i,3*Dim+1,i) = pbest.col(i);
                    if (abs(as_scalar(offsprings(Dim,i))-searched_minimum) <= abs(as_scalar(best_model(Dim))-searched_minimum)) {
                        best_model = offsprings.col(i);
                        best_simOutputs = ann.simOutputs;
                    }
                  }//end of evaluation of fittnes condition
              }//end of offspring population loop
      }//end of max number of function evaluation loop
      //cout << "POpul: \n" << POpulation << endl;
    return POpulation;
}

/**
 * Pure SC PSO/random inertia weight
 */
void pso::SC_PSO_RandomW(unsigned int ensemble_number,mlp& ann)
{
    stringstream EN_INFO1;
    EN_INFO1 << "_ens_" << ensemble_number;

    path_out_file +=  "/SC";
    path_out_file += SCPSO_NAME.c_str() +  EN_INFO.str() + EN_INFO1.str();
    initialize_all_POpulation();
//    cout << endl << path_out_file;
    unsigned int kk=0;
    while( (kk < max_number_of_shuffles)){
            sort_model_param_parents_mat();
            make_Compl_from_mat();
            for (unsigned int i =0;i<N_Complexes ; i++){
              Model_Comp_PSO.slice(i) = PSO_RandomW(Model_Comp_PSO.slice(i),ann);
            }
            make_mat_from_Compl();
            kk++;
    }//end while loop
}

/**
 * PSO/linear time varying inertia weight
 */
mat pso::PSO_LinTimeVaryingW(mat POpulation,mlp& ann)
{
    mat offsprings;
    mat velocity;

    double w1 = 0.9;  //initial inertia weight;
    double w2 = 0.4;  //final inertia weight;

    offsprings.set_size(Dim+1,n_one_population);
    velocity.set_size(Dim,n_one_population);

    offsprings.fill(44);
    velocity.fill(up_limit);

    for (unsigned int j=0;j<Number_of_generations_in_one_complex;j++ ){
        W = ((Number_of_generations_in_one_complex - (j+1))/Number_of_generations_in_one_complex) * (w1-w2) + w2;
            for (unsigned int i=0; i<n_one_population ; i++){  //n_one_population = Number_of_Members_in_one_Complex
                if (abs(as_scalar(POpulation(Dim,i))-searched_minimum) < abs(as_scalar(POpulation(3*Dim+1,i))-searched_minimum)) {
                    pbest.col(i) = POpulation.submat(0,i,Dim,i);
                }
                else {
                    pbest.col(i) = POpulation.submat(2*Dim+1,i,3*Dim+1,i);
                }

                for (unsigned int s =0;s <Dim ;s++ ){
                    velocity(s,i) = W * POpulation((s+Dim+1),i) + Cw * randu_interval()*(pbest(s,i) - POpulation(s,i)) + Cw * randu_interval() * (best_model(s) - POpulation(s,i));
                    velocity(s,i) = min(max(velocity(s,i),lo_limit),up_limit);
                    offsprings(s,i) = POpulation(s,i) + velocity(s,i);

                    if((as_scalar(offsprings(s,i))< lo_limit) || (as_scalar(offsprings(s,i))>up_limit)) {
                        offsprings(s,i) = POpulation(s,i);
                    }
                 }//end Dim loop
                 colvec help_col;
                 help_col.set_size(Dim);
                 help_col = offsprings.submat(0,i,Dim-1,i);

                 offsprings(Dim,i) = calc_fit(help_col,ann);
                 number_func_evaluation++;

                  if(abs(as_scalar(offsprings(Dim,i))-searched_minimum) < abs(as_scalar(POpulation(Dim,i))-searched_minimum)) {
                    POpulation.submat(0,i,Dim,i) = offsprings.col(i);
                    POpulation.submat(Dim+1,i,2*Dim,i) = velocity.col(i);
                    POpulation.submat(2*Dim+1,i,3*Dim+1,i) = pbest.col(i);
                    if (abs(as_scalar(offsprings(Dim,i))-searched_minimum) <= abs(as_scalar(best_model(Dim))-searched_minimum)) {
                        best_model = offsprings.col(i);
                        best_simOutputs = ann.simOutputs;
                    }
                  }//end of evaluation of fittnes condition
              }//end of offspring population loop
      }//end of max number of function evaluation loop
      //cout << "POpul: \n" << POpulation << endl;
    return POpulation;
}

/**
 * Pure SC PSO/linear time varying inertia weight
 */
void pso::SC_PSO_LinTimeVaryingW(unsigned int ensemble_number,mlp& ann)
{
    stringstream EN_INFO1;
    EN_INFO1 << "_ens_" << ensemble_number;

    path_out_file +=  "/SC";
    path_out_file += SCPSO_NAME.c_str() +  EN_INFO.str() + EN_INFO1.str();
    initialize_all_POpulation();
//    cout << endl << path_out_file;
    unsigned int kk=0;
    while( (kk < max_number_of_shuffles)){
            sort_model_param_parents_mat();
            make_Compl_from_mat();
            for (unsigned int i =0;i<N_Complexes ; i++){
              Model_Comp_PSO.slice(i) = PSO_LinTimeVaryingW(Model_Comp_PSO.slice(i),ann);
            }
            make_mat_from_Compl();
            kk++;
    }//end while loop
}

/**
 * PSO/linear time varying inertia weight with random changes
 */
mat pso::PSO_ChaoticW(mat POpulation,mlp& ann)
{
    mat offsprings;
    mat velocity;

    double w1 = 0.9;  //initial inertia weight;
    double w2 = 0.4;  //final inertia weight;
    double help_zzz = randu_interval();
    double help_z;

    offsprings.set_size(Dim+1,n_one_population);
    velocity.set_size(Dim,n_one_population);

    offsprings.fill(44);
    velocity.fill(up_limit);

    for (unsigned int j=0;j<Number_of_generations_in_one_complex;j++ ){
        help_z = 4 * help_zzz * (1 - help_zzz);
        help_zzz = help_z;
        W = (w1-w2) * (Number_of_generations_in_one_complex - (j+1))/Number_of_generations_in_one_complex + w2 * help_z;
            for (unsigned int i=0; i<n_one_population ; i++){  //n_one_population = Number_of_Members_in_one_Complex
                if (abs(as_scalar(POpulation(Dim,i))-searched_minimum) < abs(as_scalar(POpulation(3*Dim+1,i))-searched_minimum)) {
                    pbest.col(i) = POpulation.submat(0,i,Dim,i);
                }
                else {
                    pbest.col(i) = POpulation.submat(2*Dim+1,i,3*Dim+1,i);
                }

                for (unsigned int s =0;s <Dim ;s++ ){
                    velocity(s,i) = W * POpulation((s+Dim+1),i) + Cw * randu_interval()*(pbest(s,i) - POpulation(s,i)) + Cw * randu_interval() * (best_model(s) - POpulation(s,i));
                    velocity(s,i) = min(max(velocity(s,i),lo_limit),up_limit);
                    offsprings(s,i) = POpulation(s,i) + velocity(s,i);

                    if((as_scalar(offsprings(s,i))< lo_limit) || (as_scalar(offsprings(s,i))>up_limit)) {
                        offsprings(s,i) = POpulation(s,i);
                    }
                 }//end Dim loop
                 colvec help_col;
                 help_col.set_size(Dim);
                 help_col = offsprings.submat(0,i,Dim-1,i);

                 offsprings(Dim,i) = calc_fit(help_col,ann);
                 number_func_evaluation++;

                  if(abs(as_scalar(offsprings(Dim,i))-searched_minimum) < abs(as_scalar(POpulation(Dim,i))-searched_minimum)) {
                    POpulation.submat(0,i,Dim,i) = offsprings.col(i);
                    POpulation.submat(Dim+1,i,2*Dim,i) = velocity.col(i);
                    POpulation.submat(2*Dim+1,i,3*Dim+1,i) = pbest.col(i);
                    if (abs(as_scalar(offsprings(Dim,i))-searched_minimum) <= abs(as_scalar(best_model(Dim))-searched_minimum)) {
                        best_model = offsprings.col(i);
                        best_simOutputs = ann.simOutputs;
                    }
                  }//end of evaluation of fittnes condition
              }//end of offspring population loop
      }//end of max number of function evaluation loop
      //cout << "POpul: \n" << POpulation << endl;
    return POpulation;
}

/**
 * Pure SC PSO/linear time varying inertia weight with random changes
 */
void pso::SC_PSO_ChaoticW(unsigned int ensemble_number,mlp& ann)
{
    stringstream EN_INFO1;
    EN_INFO1 << "_ens_" << ensemble_number;

    path_out_file +=  "/SC";
    path_out_file += SCPSO_NAME.c_str() +  EN_INFO.str() + EN_INFO1.str();
    initialize_all_POpulation();
//    cout << endl << path_out_file;
    unsigned int kk=0;
    while( (kk < max_number_of_shuffles)){
            sort_model_param_parents_mat();
            make_Compl_from_mat();
            for (unsigned int i =0;i<N_Complexes ; i++){
              Model_Comp_PSO.slice(i) = PSO_ChaoticW(Model_Comp_PSO.slice(i),ann);
            }
            make_mat_from_Compl();
            kk++;
    }//end while loop
}

/**
 * PSO/linear time varying inertia weight with random changes
 */
mat pso::PSO_ChaoticRandW(mat POpulation,mlp& ann)
{
    mat offsprings;
    mat velocity;

    double help_z;
    double help_zzz = randu_interval();

    offsprings.set_size(Dim+1,n_one_population);
    velocity.set_size(Dim,n_one_population);

    offsprings.fill(44);
    velocity.fill(up_limit);

    for (unsigned int j=0;j<Number_of_generations_in_one_complex;j++ ){
        help_z = 4 * help_zzz * (1 - help_zzz);
        help_zzz = help_z;
        W = 0.5 * randu_interval() + 0.5 * help_z;
            for (unsigned int i=0; i<n_one_population ; i++){  //n_one_population = Number_of_Members_in_one_Complex
                if (abs(as_scalar(POpulation(Dim,i))-searched_minimum) < abs(as_scalar(POpulation(3*Dim+1,i))-searched_minimum)) {
                    pbest.col(i) = POpulation.submat(0,i,Dim,i);
                }
                else {
                    pbest.col(i) = POpulation.submat(2*Dim+1,i,3*Dim+1,i);
                }

                for (unsigned int s =0;s <Dim ;s++ ){
                    velocity(s,i) = W * POpulation((s+Dim+1),i) + Cw * randu_interval()*(pbest(s,i) - POpulation(s,i)) + Cw * randu_interval() * (best_model(s) - POpulation(s,i));
                    velocity(s,i) = min(max(velocity(s,i),lo_limit),up_limit);
                    offsprings(s,i) = POpulation(s,i) + velocity(s,i);

                    if((as_scalar(offsprings(s,i))< lo_limit) || (as_scalar(offsprings(s,i))>up_limit)) {
                        offsprings(s,i) = POpulation(s,i);
                    }
                 }//end Dim loop
                 colvec help_col;
                 help_col.set_size(Dim);
                 help_col = offsprings.submat(0,i,Dim-1,i);

                 offsprings(Dim,i) = calc_fit(help_col,ann);
                 number_func_evaluation++;

                  if(abs(as_scalar(offsprings(Dim,i))-searched_minimum) < abs(as_scalar(POpulation(Dim,i))-searched_minimum)) {
                    POpulation.submat(0,i,Dim,i) = offsprings.col(i);
                    POpulation.submat(Dim+1,i,2*Dim,i) = velocity.col(i);
                    POpulation.submat(2*Dim+1,i,3*Dim+1,i) = pbest.col(i);
                    if (abs(as_scalar(offsprings(Dim,i))-searched_minimum) <= abs(as_scalar(best_model(Dim))-searched_minimum)) {
                        best_model = offsprings.col(i);
                        best_simOutputs = ann.simOutputs;
                    }
                  }//end of evaluation of fittnes condition
              }//end of offspring population loop
      }//end of max number of function evaluation loop
      //cout << "POpul: \n" << POpulation << endl;
    return POpulation;
}

/**
 * Pure SC PSO/linear time varying inertia weight with random changes
 */
void pso::SC_PSO_ChaoticRandW(unsigned int ensemble_number,mlp& ann)
{
    stringstream EN_INFO1;
    EN_INFO1 << "_ens_" << ensemble_number;

    path_out_file +=  "/SC";
    path_out_file += SCPSO_NAME.c_str() +  EN_INFO.str() + EN_INFO1.str();
    initialize_all_POpulation();
//    cout << endl << path_out_file;
    unsigned int kk=0;
    while( (kk < max_number_of_shuffles)){
            sort_model_param_parents_mat();
            make_Compl_from_mat();
            for (unsigned int i =0;i<N_Complexes ; i++){
              Model_Comp_PSO.slice(i) = PSO_ChaoticRandW(Model_Comp_PSO.slice(i),ann);
            }
            make_mat_from_Compl();
            kk++;
    }//end while loop
}

/**
 * PSO/nonlinear time varying inertia weight
 */
mat pso::PSO_NonlinTimeConstW(mat POpulation,mlp& ann)
{
    mat offsprings;
    mat velocity;

    offsprings.set_size(Dim+1,n_one_population);
    velocity.set_size(Dim,n_one_population);

    offsprings.fill(44);
    velocity.fill(up_limit);

    double w_init = randu_interval();

    for (unsigned int j=0;j<Number_of_generations_in_one_complex;j++ ){
        W = w_init * pow(1.0002,(j+1));
            for (unsigned int i=0; i<n_one_population ; i++){  //n_one_population = Number_of_Members_in_one_Complex
                if (abs(as_scalar(POpulation(Dim,i))-searched_minimum) < abs(as_scalar(POpulation(3*Dim+1,i))-searched_minimum)) {
                    pbest.col(i) = POpulation.submat(0,i,Dim,i);
                }
                else {
                    pbest.col(i) = POpulation.submat(2*Dim+1,i,3*Dim+1,i);
                }

                for (unsigned int s =0; s<Dim ;s++ ){
                    velocity(s,i) = W * POpulation((s+Dim+1),i) + Cw * randu_interval()*(pbest(s,i) - POpulation(s,i)) + Cw * randu_interval() * (best_model(s) - POpulation(s,i));
                    velocity(s,i) = min(max(velocity(s,i),lo_limit),up_limit);
                    offsprings(s,i) = POpulation(s,i) + velocity(s,i);

                    if((as_scalar(offsprings(s,i))< lo_limit) || (as_scalar(offsprings(s,i))>up_limit)) {
                        offsprings(s,i) = POpulation(s,i);
                    }
                 }//end Dim loop
                 colvec help_col;
                 help_col.set_size(Dim);
                 help_col = offsprings.submat(0,i,Dim-1,i);

                 offsprings(Dim,i) = calc_fit(help_col,ann);
                 number_func_evaluation++;

                  if(abs(as_scalar(offsprings(Dim,i))-searched_minimum) < abs(as_scalar(POpulation(Dim,i))-searched_minimum)) {
                    POpulation.submat(0,i,Dim,i) = offsprings.col(i);
                    POpulation.submat(Dim+1,i,2*Dim,i) = velocity.col(i);
                    POpulation.submat(2*Dim+1,i,3*Dim+1,i) = pbest.col(i);
                    if (abs(as_scalar(offsprings(Dim,i))-searched_minimum) <= abs(as_scalar(best_model(Dim))-searched_minimum)) {
                        best_model = offsprings.col(i);
                        best_simOutputs = ann.simOutputs;
                    }
                  }//end of evaluation of fittnes condition
              }//end of offspring population loop
      }//end of max number of function evaluation loop
     // cout << "POpul: \n" << POpulation << endl;
    return POpulation;
}

/**
 * Pure SC PSO/nonlinear time varying inertia weight
 */
void pso::SC_PSO_NonlinTimeConstW(unsigned int ensemble_number,mlp& ann)
{
    stringstream EN_INFO1;
    EN_INFO1 << "_ens_" << ensemble_number;

    path_out_file +=  "/SC";
    path_out_file += SCPSO_NAME.c_str() +  EN_INFO.str() + EN_INFO1.str();
    initialize_all_POpulation();
//    cout << endl << path_out_file;
    unsigned int kk=0;
    while( (kk < max_number_of_shuffles)){
            sort_model_param_parents_mat();
            make_Compl_from_mat();
            for (unsigned int i =0;i<N_Complexes ; i++){
              Model_Comp_PSO.slice(i) = PSO_NonlinTimeConstW(Model_Comp_PSO.slice(i),ann);
            }
            make_mat_from_Compl();
            kk++;
    }//end while loop
}

/**
 * PSO/nonlinear time varying inertia weight
 */
mat pso::PSO_NonlinTimeVaryingW(mat POpulation,mlp& ann)
{
    mat offsprings;
    mat velocity;

    offsprings.set_size(Dim+1,n_one_population);
    velocity.set_size(Dim,n_one_population);

    offsprings.fill(44);
    velocity.fill(up_limit);

    for (unsigned int j=0;j<Number_of_generations_in_one_complex;j++ ){
        W = pow((2.0/(j+1)),0.3);
            for (unsigned int i=0; i<n_one_population ; i++){  //n_one_population = Number_of_Members_in_one_Complex
                if (abs(as_scalar(POpulation(Dim,i))-searched_minimum) < abs(as_scalar(POpulation(3*Dim+1,i))-searched_minimum)) {
                    pbest.col(i) = POpulation.submat(0,i,Dim,i);
                }
                else {
                    pbest.col(i) = POpulation.submat(2*Dim+1,i,3*Dim+1,i);
                }

                for (unsigned int s =0;s <Dim ;s++ ){
                    velocity(s,i) = W * POpulation((s+Dim+1),i) + Cw * randu_interval()*(pbest(s,i) - POpulation(s,i)) + Cw * randu_interval() * (best_model(s) - POpulation(s,i));
                    velocity(s,i) = min(max(velocity(s,i),lo_limit),up_limit);
                    offsprings(s,i) = POpulation(s,i) + velocity(s,i);

                    if((as_scalar(offsprings(s,i))< lo_limit) || (as_scalar(offsprings(s,i))>up_limit)) {
                        offsprings(s,i) = POpulation(s,i);
                    }
                 }//end Dim loop
                 colvec help_col;
                 help_col.set_size(Dim);
                 help_col = offsprings.submat(0,i,Dim-1,i);

                 offsprings(Dim,i) = calc_fit(help_col,ann);
                 number_func_evaluation++;

                  if(abs(as_scalar(offsprings(Dim,i))-searched_minimum) < abs(as_scalar(POpulation(Dim,i))-searched_minimum)) {
                    POpulation.submat(0,i,Dim,i) = offsprings.col(i);
                    POpulation.submat(Dim+1,i,2*Dim,i) = velocity.col(i);
                    POpulation.submat(2*Dim+1,i,3*Dim+1,i) = pbest.col(i);
                    if (abs(as_scalar(offsprings(Dim,i))-searched_minimum) <= abs(as_scalar(best_model(Dim))-searched_minimum)) {
                        best_model = offsprings.col(i);
                        best_simOutputs = ann.simOutputs;
                    }
                  }//end of evaluation of fittnes condition
              }//end of offspring population loop
      }//end of max number of function evaluation loop
      //cout << "POpul: \n" << POpulation << endl;
    return POpulation;
}

/**
 * Pure SC PSO/nonlinear time varying inertia weight
 */
void pso::SC_PSO_NonlinTimeVaryingW(unsigned int ensemble_number,mlp& ann)
{
    stringstream EN_INFO1;
    EN_INFO1 << "_ens_" << ensemble_number;

    path_out_file +=  "/SC";
    path_out_file += SCPSO_NAME.c_str() +  EN_INFO.str() + EN_INFO1.str();
    initialize_all_POpulation();
//    cout << endl << path_out_file;
    unsigned int kk=0;
    while( (kk < max_number_of_shuffles)){
            sort_model_param_parents_mat();
            make_Compl_from_mat();
            for (unsigned int i =0;i<N_Complexes ; i++){
              Model_Comp_PSO.slice(i) = PSO_NonlinTimeVaryingW(Model_Comp_PSO.slice(i),ann);
            }
            make_mat_from_Compl();
            kk++;
    }//end while loop
}

/**
 * PSO/adaptive inertia weight
 */
mat pso::PSO_AdaptW(mat POpulation,mlp& ann)
{
    mat offsprings;
    mat velocity;

    double w_max = 1;
    double w_min = 0;

    double success_count;
    double probab;
    W = 1;

    offsprings.set_size(Dim+1,n_one_population);
    velocity.set_size(Dim,n_one_population);

    offsprings.fill(44);
    velocity.fill(up_limit);


    for (unsigned int m=0; m<n_one_population ; m++){
        pbest.col(m) = POpulation.submat(0,m,Dim,m);
    }

    for (unsigned int j=0;j<Number_of_generations_in_one_complex;j++ ){
            success_count = 0;
            for (unsigned int i=0; i<n_one_population ; i++){  //n_one_population = Number_of_Members_in_one_Complex
                for (unsigned int s =0;s <Dim ;s++ ){
                    velocity(s,i) = W * POpulation((s+Dim+1),i) + Cw * randu_interval()*(pbest(s,i) - POpulation(s,i)) + Cw * randu_interval() * (best_model(s) - POpulation(s,i));
                    velocity(s,i) = min(max(velocity(s,i),lo_limit),up_limit);
                    offsprings(s,i) = POpulation(s,i) + velocity(s,i);

                    if((as_scalar(offsprings(s,i))< lo_limit) || (as_scalar(offsprings(s,i))>up_limit)) {
                        offsprings(s,i) = POpulation(s,i);
                    }
                 }//end Dim loop
                 colvec help_col;
                 help_col.set_size(Dim);
                 help_col = offsprings.submat(0,i,Dim-1,i);

                 offsprings(Dim,i) = calc_fit(help_col,ann);
                 number_func_evaluation++;

                  if(abs(as_scalar(offsprings(Dim,i))-searched_minimum) < abs(as_scalar(pbest(Dim,i))-searched_minimum)) {
                    pbest.col(i) = offsprings.col(i);
                    success_count = success_count + 1;
                  }

                  if(abs(as_scalar(offsprings(Dim,i))-searched_minimum) < abs(as_scalar(POpulation(Dim,i))-searched_minimum)) {
                    POpulation.submat(0,i,Dim,i) = offsprings.col(i);
                    POpulation.submat(Dim+1,i,2*Dim,i) = velocity.col(i);
                    POpulation.submat(2*Dim+1,i,3*Dim+1,i) = pbest.col(i);

                    if (abs(as_scalar(offsprings(Dim,i))-searched_minimum) <= abs(as_scalar(best_model(Dim))-searched_minimum)) {
                        best_model = offsprings.col(i);
                        best_simOutputs = ann.simOutputs;
                    }
                  }//end of evaluation of fittnes condition
              }//end of offspring population loop
                //cout << "\nsuccess: " << success_count << endl;
                //cout << "\nn_one_pop: " << n_one_population << endl;
              probab = success_count / n_one_population;
                //cout << "\nprobab: " << probab << endl;
              W = (w_max - w_min) * probab + w_min;
              //cout << "\nW: " << W << endl;
     }//end of max number of function evaluation loop
    return POpulation;
}

/**
 * Pure SC PSO/adaptive inertia weight
 */
void pso::SC_PSO_AdaptW(unsigned int ensemble_number,mlp& ann)
{
    stringstream EN_INFO1;
    EN_INFO1 << "_ens_" << ensemble_number;

    path_out_file +=  "/SC";
    path_out_file += SCPSO_NAME.c_str() +  EN_INFO.str() + EN_INFO1.str();
    initialize_all_POpulation();
//    cout << endl << path_out_file;
    unsigned int kk=0;
    while( (kk < max_number_of_shuffles)){
            sort_model_param_parents_mat();
            make_Compl_from_mat();
            for (unsigned int i =0;i<N_Complexes ; i++){
              Model_Comp_PSO.slice(i) = PSO_AdaptW(Model_Comp_PSO.slice(i),ann);
            }
            make_mat_from_Compl();
            kk++;
    }//end while loop
}

/**
 * PSO/adaptive inertia weight for each particle
 */
mat pso::PSO_AdaptParticleW(mat POpulation,mlp& ann)
{
    mat offsprings;
    mat velocity;

    double wMax = 0.9;  //initial inertia weight;
    double wMin = 0.1;  //final inertia weight;
    colvec WP;

    offsprings.set_size(Dim+1,n_one_population);
    velocity.set_size(Dim,n_one_population);
    WP.set_size(n_one_population);

    offsprings.fill(44);
    velocity.fill(up_limit);
    WP.fill(0.9);

    for (unsigned int j=0;j<Number_of_generations_in_one_complex;j++ ){
            for (unsigned int i=0; i<n_one_population ; i++){  //n_one_population = Number_of_Members_in_one_Complex
                if (abs(as_scalar(POpulation(Dim,i))-searched_minimum) < abs(as_scalar(POpulation(3*Dim+1,i))-searched_minimum)) {
                    pbest.col(i) = POpulation.submat(0,i,Dim,i);
                    WP(i) = ((wMax+wMin)/2.0 - wMin) * randu_interval() + wMin;
                }
                else {
                    pbest.col(i) = POpulation.submat(2*Dim+1,i,3*Dim+1,i);
                    WP(i) = ((wMax+wMin)/2.0 - wMin) * randu_interval() + ((wMax+wMin)/2.0);
                }

                for (unsigned int s =0;s <Dim ;s++ ){
                    velocity(s,i) = WP(i) * POpulation((s+Dim+1),i) + Cw * randu_interval()*(pbest(s,i) - POpulation(s,i)) + Cw * randu_interval() * (best_model(s) - POpulation(s,i));
                    velocity(s,i) = min(max(velocity(s,i),lo_limit),up_limit);
                    offsprings(s,i) = POpulation(s,i) + velocity(s,i);

                    if((as_scalar(offsprings(s,i))< lo_limit) || (as_scalar(offsprings(s,i))>up_limit)) {
                        offsprings(s,i) = POpulation(s,i);
                    }
                 }//end Dim loop
                 colvec help_col;
                 help_col.set_size(Dim);
                 help_col = offsprings.submat(0,i,Dim-1,i);

                 offsprings(Dim,i) = calc_fit(help_col,ann);
                 number_func_evaluation++;

                  if(abs(as_scalar(offsprings(Dim,i))-searched_minimum) < abs(as_scalar(POpulation(Dim,i))-searched_minimum)) {
                    POpulation.submat(0,i,Dim,i) = offsprings.col(i);
                    POpulation.submat(Dim+1,i,2*Dim,i) = velocity.col(i);
                    POpulation.submat(2*Dim+1,i,3*Dim+1,i) = pbest.col(i);
                    if (abs(as_scalar(offsprings(Dim,i))-searched_minimum) <= abs(as_scalar(best_model(Dim))-searched_minimum)) {
                        best_model = offsprings.col(i);
                        best_simOutputs = ann.simOutputs;
                    }
                  }//end of evaluation of fittnes condition
              }//end of offspring population loop
      }//end of max number of function evaluation loop
      //cout << "POpul: \n" << POpulation << endl;
    return POpulation;
}

/**
 * Pure SC PSO/adaptive inertia weight for each particle
 */
void pso::SC_PSO_AdaptParticleW(unsigned int ensemble_number,mlp& ann)
{
    stringstream EN_INFO1;
    EN_INFO1 << "_ens_" << ensemble_number;

    path_out_file +=  "/SC";
    path_out_file += SCPSO_NAME.c_str() +  EN_INFO.str() + EN_INFO1.str();
    initialize_all_POpulation();
//    cout << endl << path_out_file;
    unsigned int kk=0;
    while( (kk < max_number_of_shuffles)){
            sort_model_param_parents_mat();
            make_Compl_from_mat();
            for (unsigned int i =0;i<N_Complexes ; i++){
              Model_Comp_PSO.slice(i) = PSO_AdaptParticleW(Model_Comp_PSO.slice(i),ann);
            }
            make_mat_from_Compl();
            kk++;
    }//end while loop
}


/**
 * The general interface to the problem testings
 PSO_selection_0-ConstrFactor_1-ConstantW_2-RandomW_3-LinTimeVaryingW_4-ChaoticW_5-ChaoticRandW_
 PSO_selection_6-NonlinTimeConstW_7-NonlinTimeVaryingW_8-AdaptW_9-AdaptParticleW
 */
void pso::run_ensemble(mlp& ann)
{
    cout << "\n\n**** ENSEMBLE RUN ****\n";
    /* mj
    string filet_names; //file with names of ensemble outcomes

    filet_names +=directory + "/File_names" + SCPSO_NAME.c_str() + EN_INFO.str();

    ofstream filet_name_stream(filet_names.c_str(), ios::app);
      if (!filet_name_stream) {
       cout << "\nIt is impossible to write to file with files names " << filet_name_stream;
       exit(EXIT_FAILURE);
     }
     */
     best_model_ens_final = -10e8;
     best_simOutputs_fin.fill(10e8);
     best_weights_fin.fill(10e8);
     best_best_simOutputs_fin.fill(10e8);
     best_best_weights_fin.fill(10e8);

    for(unsigned int i=0; i<ensemble;i++){
            srand(i+1);
            if (!path_out_file.empty()) path_out_file.clear();
            path_out_file += directory;
            cout << "\nEnsemble simulation: " << i+1;
            number_func_evaluation = 0;
            switch (SCPSO_selection)
            {
                case ConstrFactor:
                    SC_PSO_ConstrFactor(i,ann);
                    break;
                case ConstantW:
                    SC_PSO_ConstantW(i,ann);
                    break;
                case RandomW:
                    SC_PSO_RandomW(i,ann);
                    break;
                case LinTimeVaryingW:
                    SC_PSO_LinTimeVaryingW(i,ann);
                    break;
                case ChaoticW:
                    SC_PSO_ChaoticW(i,ann);
                    break;
                case ChaoticRandW:
                    SC_PSO_ChaoticRandW(i,ann);
                    break;
                case NonlinTimeConstW:
                    SC_PSO_NonlinTimeConstW(i,ann);
                    break;
                case NonlinTimeVaryingW:
                    SC_PSO_NonlinTimeVaryingW(i,ann);
                    break;
                case AdaptW:
                    SC_PSO_AdaptW(i,ann);
                    break;
                case AdaptParticleW:
                    SC_PSO_AdaptParticleW(i,ann);
                    break;
                 default:
                    break;
            }
            //mj filet_name_stream << path_out_file.c_str() << endl;
            cout << endl <<"Best model: " << as_scalar(best_model(Dim)) << endl;

            best_model_ens(i) = as_scalar(best_model(Dim));
            best_weights =  best_model.subvec(0,Dim-1);

            best_simOutputs_fin.col(i) = best_simOutputs;
            best_weights_fin.col(i) = best_weights;

//            if ((PRoblem_Number == 0) || (PRoblem_Number == 1) || (PRoblem_Number == 2) || (PRoblem_Number == 5) || (PRoblem_Number == 6) || (PRoblem_Number == 8) || (PRoblem_Number == 9)  || (PRoblem_Number == 10) ) {
//                if (abs(best_model_ens(i)) < abs(best_model_ens_final)) {
//                    best_model_ens_final = best_model_ens(i);
//                    best_best_simOutputs_fin = best_simOutputs_fin.col(i);
//                    best_best_weights_fin = best_weights_fin.col(i);
//                }
//            }
//            else if ((PRoblem_Number == 3) || (PRoblem_Number == 4) || (PRoblem_Number == 7)) {
//                if (best_model_ens(i) > best_model_ens_final) {
//                    best_model_ens_final = best_model_ens(i);
//                    best_best_simOutputs_fin = best_simOutputs_fin.col(i);
//                    best_best_weights_fin = best_weights_fin.col(i);
//                }
//            }

                if (abs(best_model_ens(i)) < abs(best_model_ens_final)) {
                    best_model_ens_final = best_model_ens(i);
                    best_best_simOutputs_fin = best_simOutputs_fin.col(i);
                    best_best_weights_fin = best_weights_fin.col(i);
                }

            ann.out_fittness(best_model_ens(i),(i+1),PRoblem_Number, "./data/outputs/ok_ens");

            best_model.fill(10e8);
            best_simOutputs.fill(10e8);
            best_weights.fill(10e8);
        }

        ann.out_outputs(best_best_simOutputs_fin,"./data/outputs/best_best_simOutputs");
        ann.out_outputs(best_best_weights_fin,"./data/outputs/best_best_weights");

        ann.out_outputs(best_simOutputs_fin,"./data/outputs/best_simOutputs");
        ann.out_outputs(best_weights_fin,"./data/outputs/best_weights");

        ann.out_fittness(best_model_ens_final,99,PRoblem_Number, "./data/outputs/ok_cal");
       // best_model_ens_final = min(best_model_ens) ;
        cout << "\nBest Best Best: " << best_model_ens_final << endl;

        //mj filet_name_stream.close();

        cout << endl << "\n***The ensemble run has been finished.***" << endl;

       //mj calc_quantiles_fast_small();
       //mj cout << endl << "***The quantiles has been computed.***\n" << endl;

}

/**
 * Removes the duplicated elements in my_vec
 the implementation for type 7 of estimation of quantiles
TODO implementa rest quantile estimators
*/
template<class my_vec>  my_vec pso::unique_(my_vec data)
{
   data = sort(data);

   my_vec return_vec;
   return_vec.set_size(data.n_elem);
   return_vec.fill(999999);

   unsigned int number_d =0, help_ind =0, help_var =0;
   number_d = data.n_elem;

    for (unsigned int i=0;i< number_d;i++ ){
           while (as_scalar(data(help_ind)) == as_scalar(data(help_var)) ){
               help_var++;
           }
           help_ind += (help_var - help_ind) + 1;
           help_var = help_ind;
           return_vec(help_ind) = data(help_ind);
      }

   return_vec.reshape(help_ind);

   return(return_vec);
}

/**
 * Returns vector of quantiles for given data

  Discontinuous:
       INV_emp_dist - inverse of the empirical distribution function

 NOT IMPLEMENTED
  //   INV_emp_dist_av - like type 1, but with averaging at discontinuities (g=0)
  //   SAS_near_int - SAS definition: nearest even order statistic
   //  Piecwise linear continuous:
   //    In this case, sample quantiles can be obtained by linear interpolation
   //    between the k-th order statistic and p(k).
   //    type=4 - linear interpolation of empirical cdf, p(k)=k/n;
   //    type=5 - hydrolgical a very popular definition, p(k) = (k-0.5)/n;
   //    type=6 - used by Minitab and SPSS, p(k) = k/(n+1);
   //    type=7 - used by S-Plus and R, p(k) = (k-1)/(n-1);
   //    type=8 - resulting sample quantiles are approximately median unbiased
   //             regardless of the distribution of x. p(k) = (k-1/3)/(n+1/3);
   //    type=9 - resulting sample quantiles are approximately unbiased, when
   //             the sample comes from Normal distribution. p(k)=(k-3/8)/(n+1/4);
   //
   //    default type = 7
   //
    References:
    1) Hyndman, R.J and Fan, Y, (1996) "Sample quantiles in statistical packages"
                                        American Statistician, 50, 361-365

*/
template<class my_vec> my_vec  pso::quantiles_(my_vec data, my_vec probs)
{
    my_vec sorted_data;
    my_vec quantile;

    sorted_data = sort(data);
    quantile.set_size(probs.n_elem);

    uvec index;
    index.set_size(probs.n_elem);

    switch (quantile_selection)
    {
        case INV_emp_dist:
            for (unsigned int i=0; i<probs.n_elem  ;i++ ){
                    if(as_scalar(probs(i)) == 0.0) index(i) = 0;
                        else index(i) = static_cast<unsigned int> (ceil((as_scalar(probs(i) * sorted_data.n_elem)))) -1;
                   }
            for (unsigned int i=0; i<probs.n_elem  ;i++ ){
                      quantile(i) = sorted_data(index(i));
                  }
            break;

//        case INV_emp_dist_av:
//
//            break;
//        default:
//            break;
    }

 //   cout << index;
    return(quantile);
}

/**
 * Slow Quantile estimation for large problems
 */
void pso::calc_quantiles()
{
/*     working example */
//     colvec data;
//     data << 1<< 2<< 3<< 4<< 5 << 6 << 7 << 8<< 9 << 10 <<11 << 12 << 13 << 14 << 15 << 16 << 17 << 18 << 19 << 20 << 21 << 22 << 23 << 24 << 25;
     rowvec prst;
     prst << 0.<< 0.25<< 0.5 << 0.75 << 1.0;

//     cout << data;
//     cout << endl << quantiles_(data, prst);
   string filet_names;//file with names of ensemble outcomes

    filet_names +=directory + "/File_names" + SCPSO_NAME.c_str() + EN_INFO.str();
    //filet_names += directory + "/File_names" + "_SC" + SCPSO_NAME.c_str() + EN_INFO.str();

 //   cout << filet_names;

    string *file_en, trash;
    unsigned int number_of_files = 0;

    ifstream pso_file_ens_names(filet_names.c_str());
    while(pso_file_ens_names.good()){
    pso_file_ens_names >> trash;
    number_of_files++;

    }
    number_of_files--;
    cout << endl << "Opening the "<< number_of_files << endl;
    file_en = new string[number_of_files];

    pso_file_ens_names.clear();
    pso_file_ens_names.seekg (0, ios::beg);

    for (unsigned int i =0; i< number_of_files ;i++ ){
      pso_file_ens_names >> file_en[i] ;
    //  cout << file_en[i] << endl;
      }
    pso_file_ens_names.close();

    ifstream first_filet(file_en[0].c_str());
    unsigned int num_rows =0;
    while(first_filet.good()){
        getline(first_filet,trash);
        num_rows++;
        }
    first_filet.close();
    num_rows --;

    rowvec Quantiles_ens;
    Quantiles_ens.set_size(prst.n_elem+2);

    rowvec help_vec1;
    mat  data_one_file;

    string filet_quantiles;
    filet_quantiles +=directory + "/Ensemble_quantiles_SC_" +SCPSO_NAME.c_str() + EN_INFO.str();

    ofstream out_quantile_stream(filet_quantiles.c_str(), ios::app);
      if (!out_quantile_stream) {
       cout << "\nIt is impossible to write to file  " << filet_quantiles;
       exit(EXIT_FAILURE);
      }

    cout << "\nComputing the Quantiles\n";
    for (unsigned int j=0; j<num_rows ; j++){
      help_vec1.set_size(number_of_files);
      for (unsigned int i =0; i<number_of_files; i++){
             data_one_file.load(file_en[i].c_str());
             help_vec1(i) = data_one_file(j,1);
             if(j==0) cout <<  i;
             data_one_file.reset();
             }
             cout << endl<< help_vec1;
      Quantiles_ens.subvec(0,prst.n_elem-1) = quantiles_<rowvec>(help_vec1, prst);
      Quantiles_ens(prst.n_elem) = mean(help_vec1);
      Quantiles_ens(prst.n_elem+1) = stddev(help_vec1);
      for(unsigned int qu =0; qu < (prst.n_elem+2); qu++){
        out_quantile_stream << Quantiles_ens(qu) << "\t";
      }
      out_quantile_stream << "\n";
      help_vec1.reset();
     }

    delete [] file_en;
    out_quantile_stream.close();
}

/**
 * Fast Quantile estimation for middle problems
 */
void pso::calc_quantiles_fast_small()
{
/*     working example */
//     colvec data;
//     data << 1<< 2<< 3<< 4<< 5 << 6 << 7 << 8<< 9 << 10 <<11 << 12 << 13 << 14 << 15 << 16 << 17 << 18 << 19 << 20 << 21 << 22 << 23 << 24 << 25;
     rowvec prst;
     prst << 0.0 << 0.25<< 0.5 << 0.75 << 1.0;

//     cout << data;
//     cout << endl << quantiles_(data, prst);
   string filet_names;//file with names of ensemble outcomes

    filet_names +=directory + "/File_names" + SCPSO_NAME.c_str() + EN_INFO.str();
    //filet_names += directory + "/File_names" + "_SC" + SCPSO_NAME.c_str() + EN_INFO.str();

 //   cout << filet_names;

    string *file_en, trash;
    unsigned int number_of_files = 0;

    ifstream pso_file_ens_names(filet_names.c_str());
    while(pso_file_ens_names.good()){
       pso_file_ens_names >> trash;
       number_of_files++;
      }
    number_of_files--;
//    cout << endl << "Opening the "<< number_of_files << endl;
    file_en = new string[number_of_files];

    pso_file_ens_names.clear();
    pso_file_ens_names.seekg (0, ios::beg);

    for (unsigned int i =0; i< number_of_files ;i++ ){
      pso_file_ens_names >> file_en[i] ;
    //  cout << file_en[i] << endl;
      }
    pso_file_ens_names.close();

    ifstream first_filet(file_en[0].c_str());
    unsigned int num_rows =0;
    while(first_filet.good()){
        getline(first_filet,trash);
        num_rows++;
        }
    first_filet.close();
    num_rows --;
   // cout << num_rows << endl;

    mat Quantiles_ens;
    Quantiles_ens.set_size(num_rows, prst.n_elem+2);



    mat  data_all_file;
    data_all_file.set_size(num_rows,number_of_files);

      for (unsigned int i =0; i<number_of_files; i++){
             mat data_one_file;
             data_one_file.load(file_en[i].c_str());
             data_all_file.submat(0,i,num_rows-1,i) = data_one_file.col(1);
             }

      //       cout <<"\nAll mat finallize.d\n";
    string filet_quantiles;
    filet_quantiles +=directory + "/Ensemble_quantiles_SC_" +SCPSO_NAME.c_str() + EN_INFO.str();

//    ofstream out_quantile_stream(filet_quantiles.c_str(), ios::app);
//      if (!out_quantile_stream) {
//       cout << "\nIt is impossible to write to file  " << filet_quantiles;
//       exit(EXIT_FAILURE);
//      }

    cout << "\nComputing the Quantiles\n";
    for (unsigned int j=0; j<num_rows ; j++){
      rowvec help_vec1;
      help_vec1.set_size(number_of_files);
      help_vec1 = data_all_file.row(j);
      Quantiles_ens.submat(j,0,j,prst.n_elem-1) = quantiles_<rowvec>(help_vec1, prst);
      Quantiles_ens(j,prst.n_elem) = mean(help_vec1);
      Quantiles_ens(j,prst.n_elem+1) = stddev(help_vec1);
     }

    Quantiles_ens.save(filet_quantiles.c_str(),raw_ascii);
    delete [] file_en;
//    out_quantile_stream.close();
}
