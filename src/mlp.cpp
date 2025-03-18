#include "mlp.h"
#include "pso.h"


/**
 * The constructor
 * \param settings_file: file with MLP settings
 */
mlp::mlp(string settings_file)
{
    //ctor
    string trash;

    ifstream ann_inform(settings_file.c_str());
    if(!ann_inform) { // osetreni jestli je soubor mozne otevrit
      cout << "\nUnable to open file :"<< settings_file.c_str() <<" with initial settings for computations.";
      exit(EXIT_FAILURE);
    }
    ann_inform >> scientific;
    ann_inform >> trash >> trash >> nInputs >> trash >> nHLneurons;
    ann_inform >> trash >> Pr_Number;
    ann_inform >> trash >> directoryIn >> trash >> directoryOut;
    ann_inform.close();

    //cout << "nInputs " << nInputs << " nHLneurons " << nHLneurons << endl;
    //cout << "directoryIn " << directoryIn << " directoryOut " << directoryOut << endl;

    nWeights = nInputs * nHLneurons + nHLneurons + 1;
    nTargets = 0;

    weights.set_size(nWeights); //(nInputs * nHLneurons + nHLneurons + 1);
    weights.fill(1);

    activations.set_size(nHLneurons + 1);
    activations.fill(1000);

    neuronOutputs.set_size(nHLneurons);
    neuronOutputs.fill(1000);

    Inputs.set_size(nTargets,nInputs);
    Inputs.fill(1000);

    obsOutputs.set_size(nTargets);
    obsOutputs.fill(1000);

    simOutputs.set_size(nTargets);
    simOutputs.fill(1000);

    //cout << "nInputs " << nInputs << " nTargets " << nTargets << endl;
    //cout << "Inputs " << Inputs << endl;
}

/**
 * The destructor
 */
mlp::~mlp()
{
    //dtor
}

/**
 * The copy constructor
 */
mlp::mlp(const mlp& other)
{
    //copy ctor
    nInputs = other.nInputs;
    nHLneurons = other.nHLneurons;
    Pr_Number = other.Pr_Number;
    directoryIn = other.directoryIn;
    directoryOut = other.directoryOut;
    nWeights = other.nWeights;
    nTargets = other.nTargets;
    weights = other.weights;
    activations = other.activations;
    neuronOutputs = other.neuronOutputs;
    Inputs = other.Inputs;
    obsOutputs = other.obsOutputs;
    simOutputs = other.simOutputs;
}

/**
 * The assignment operator
 */
mlp& mlp::operator=(const mlp& other)
{
    if (this == &other) {
            return *this; // handle self assignment
    }  else {
        nInputs = other.nInputs;
        nHLneurons = other.nHLneurons;
        Pr_Number = other.Pr_Number;
        directoryIn = other.directoryIn;
        directoryOut = other.directoryOut;
        nWeights = other.nWeights;
        nTargets = other.nTargets;
        weights = other.weights;
        activations = other.activations;
        neuronOutputs = other.neuronOutputs;
        Inputs = other.Inputs;
        obsOutputs = other.obsOutputs;
        simOutputs = other.simOutputs;
    }
    return *this;
}


/**
 * Create input data to the final MLP from outputs from each MLP
 * \param file_name: file with input and output data
 * \param save_file: file for saving
 */
void mlp::create_input(string load_file, string file_name, string save_file)
{
  //!! checking if the file exists
  ifstream File_name_in(file_name.c_str());
  if(!File_name_in){
    cout << "Can not open the file " << file_name << endl;
    exit(EXIT_FAILURE);
  }
  File_name_in.close();

  //!! read data
  colvec help_Inputs_fin;
  help_Inputs_fin.load(file_name);

  mat help_Outputs_fin;
  help_Outputs_fin.load(load_file);

  nTargets = help_Outputs_fin.n_rows;
  obsOutputs = help_Outputs_fin.col(help_Outputs_fin.n_cols-1);

  //!! create input from the file
  mat help_create;
  help_create.set_size(nTargets,nInputs+1);  // ncols = (nInputs) + 1 = (number of mlp models + bias) + obsOutput
  help_create.fill(1);  // the second last column is bias, therefore fill all with 1

  for (unsigned int i=0; i<(nInputs-1); i++) {
    help_create.col(i) = help_Inputs_fin.rows(i*nTargets,((i+1)*nTargets-1));
  }
  help_create.col(nInputs) = obsOutputs;  // the last column in obsOutput

  //!! save data
  //help_create.save(save_file, raw_ascii);
  ofstream out_stream(save_file.c_str()); //, ios::app);
    if (!out_stream) {
      cout << "\nIt is impossible to write to file  " << save_file;
      exit(EXIT_FAILURE);
    }
  out_stream << help_create ;
  out_stream.close();

}


/**
 * Read measured input and output data from file
 * \param file_name: file with measured input and output data
 */
void mlp::read_file(string file_name)
{
  //!! checking if the file exists
  ifstream File_name_in(file_name.c_str());
  if(!File_name_in){
    cout << "Can not open the file " << file_name << endl;
    exit(EXIT_FAILURE);
  }
  File_name_in.close();

  //!! read data
  mat help_Inputs;
  help_Inputs.load(file_name);
//cout << "help_In = " << help_Inputs << endl;

  //!! count number of rows and columns
  unsigned nrow = 0;
  unsigned ncol = 0;
  nrow = help_Inputs.n_rows;
  ncol = help_Inputs.n_cols;
//cout << "nRow = " << nrow << endl;
//cout << "nCol = " << ncol << endl;

  //!! fill data into the variables Inputs, obsOutputs, nTargets
  Inputs = help_Inputs.submat(0,0,(nrow-1),(ncol-2));
  obsOutputs = help_Inputs.col(ncol-1);
//cout << "Inputs = " << Inputs << endl;
//cout << "obsOutputs " << obsOutputs << endl;
  nTargets = nrow;
//cout << "nTargets = " << nTargets << endl;

  //!! zero inputs
  if (nInputs <= 0 || size(Inputs,1) <= 0) {
    cout << "\nNumber of inputs is equal to, or less than 0." << endl;
    exit(EXIT_FAILURE);
  }

  //!! compare the number of inputs from the setting file (nInputs) with nCols in Inputs
//cout << "nInputs = " << nInputs << endl;
//cout << "size In = " << size(Inputs,1) << endl;
  if (nInputs != size(Inputs,1)) {  // one column is bias, one column is obsOutputs
    cout << "\nNumber of columns in input data file is not the same as nInputs in setting_ann file." << endl;
    exit(EXIT_FAILURE);
  }

  //!! set the size of simOutputs according to nTargets
  simOutputs.set_size(nTargets);

}


/**
 * Read weights from file
 * \param file_name: file with weights for validation
 * \param number_model: number of mlp model (because the weights are in one file, each column for each model)
 */
void mlp::read_weights(string file_name, unsigned int number_model)
{
  unsigned int up = 0;
  unsigned int down = 0;

  colvec help_weights;
  //help_weights.set_size(nInputs*);

  //!! checking if the file exists
  ifstream File_name_in(file_name.c_str());
  if(!File_name_in){
    cout << "Can not open the file " << file_name << endl;
    exit(EXIT_FAILURE);
  }
  File_name_in.close();

  //!! read data
  help_weights.load(file_name);


  if (number_model == 99) {  // fin model
    down = size(help_weights,0) - 1;
    up = down - nWeights + 1;
    weights = help_weights.rows(up,down);
  }
  else {  // every other mlp
    up = (number_model-1) * nWeights;
    down = up + nWeights - 1;
    weights = help_weights.rows(up,down);
  }

  //cout << "up = " << up << endl;
  //cout << "down = " << down << endl;
}


/**
 * Transformation of the original data
 */
void mlp::orig_to_trans()
{
  double gam = -0.015;

  trans_Inputs = 1.0 - exp(gam * Inputs);

}



/**
 * Transformation back to the original data
 * \param trans_data: value for transformation back to the original value
 * \return value: transformed value of the trans_data
 */
double mlp::trans_to_orig(double trans_data)
{
  double value = 0.0;
  double gam = 0.015;

  if ((1-trans_data) == 0) {
      value = 1.0/gam* log(10e-30);
  }
  else {
      value = 1.0/gam * log(1/(1-trans_data));
  }

  //cout << "Inputs.row(1)" << Inputs.row(1) << endl;
  //cout << "trans_Inputs.row(1)" << trans_Inputs.row(1) << endl;

  return value;

}


/**
 * Method for calculation of the activation
 * \param help_weights: colvec of the weights for calculating of the activation
 * \param help_inputs: rowvec of the inputs for calculating of the activation
 */
double mlp::calc_output(colvec help_weights, rowvec help_inputs)
{
//  setWeights();
//cout << "weights = " << weights << endl;

  unsigned up = 0;
  unsigned down = (nInputs-1);

  //!! activation of neurons in the hidden layer (i.e. inputs+bias)
  for (unsigned int i=0; i<nHLneurons; i++) {
      activations(i) = as_scalar(help_inputs * help_weights.subvec(up,down));
      //activations(i) = as_scalar(Inputs.row(0) * weights.subvec(up,down));
      up = down+1;
      down = up + nInputs - 1;
  }

  //!! activation of the output neuron (i.e. HLneurons+bias)
  //!! transformation of the activation in HL neurons
  colvec help_activations;  //input to the transformation function, size=nHLneurons (i.e. smaller than the activation with size=nHLneurons+1)
  help_activations.set_size(nHLneurons);
  help_activations = activations.subvec(0,(nHLneurons-1));
//cout << "\nhelp_activations = " << help_activations << endl;

  calc_output_HL(help_activations);
//cout << "neuronOutputs = " << neuronOutputs << endl;

  colvec help_neuronOut;  //input for calculation of the activation of the output neuron
  help_neuronOut.set_size(nHLneurons+1);
  help_neuronOut.fill(1);
  help_neuronOut.subvec(0,(nHLneurons-1)) = neuronOutputs;  // outputs from HL neurons + bias equal to 1
//cout << "help_neuronOut = " << help_neuronOut << endl;
//cout << "help_weights = " << help_weights << endl;

  down = up + nHLneurons;
  activations(nHLneurons) = as_scalar(trans(help_neuronOut) * help_weights.subvec(up,down));
//cout << "activations = " << activations << endl;

  //!! transformed output from the MLP
  double out;
//  switch (type_trans_f) {
//    case Sigmoid:
//        out = 1 / (1 + exp(-1*activations(nHLneurons)));  // logistic sigmoid
//        break;
//    case RootSig:
//        out = activations(nHLneurons) / (1 + sqrt(1 + pow(activations(nHLneurons),2)));
//        break;
//    case Long:
//        out = 4 * (1 - 2 * exp(-0.7 * exp(activations(nHLneurons))));
//        break;
//    default:
//        break;
//  }
  out = activations(nHLneurons) / (1 + sqrt(1 + pow(activations(nHLneurons),2)));

  return out;

}


/**
 * Method for calculation of the neuron output after activation using chosen transformation function
 * \param help_activ: colvec with activations which are transformed
 */
void mlp::calc_output_HL(colvec help_activ)
{
//  switch (type_trans_f) {
//    case Sigmoid:
//        neuronOutputs = 1 / (1 + exp(-1*help_activ));  // logistic sigmoid
//        break;
//    case RootSig:
//        neuronOutputs = help_activ / (1 + sqrt(1 + pow(help_activ,2)));
//        break;
//    case Long:
//        neuronOutputs = 4 * (1 - 2 * exp(-0.7 * exp(help_activ)));
//        break;
//    default:
//        break;
//  }

      neuronOutputs = help_activ / (1 + sqrt(1 + pow(help_activ,2)));

}


/**
 * The fittnes calculation
 * \param crit: number of the calculated objective critetion
 * \param var_obs: colvec of the observed values
 * \param var_mod: colvec of the simulated values
 * \return ok: calculated fittness value
 */
double mlp::calc_fittnes(unsigned int crit, colvec var_obs, colvec var_mod)
{
  double cit = 0, jmen = 0;
  double ok = 0, ok1 = 0, ok2 = 0;
  double ww = 0.85;  // weight for criteria in PID index

  colvec d_obs, d_mod;
  d_obs.set_size(size(var_obs,0)-1);
  d_mod.set_size(size(var_mod,0)-1);

  switch (crit) {
      case 0: //MSE:
        ok = mean(pow((var_obs - var_mod), 2)); //mean squared error
        break;
      case 1: //MAE:
        ok = mean(abs(var_obs - var_mod)); //mean absolute error
        break;
      case 2: //MAPE:
        ok = mean(abs(var_obs - var_mod) / var_obs); //mean absolute percentage error
        break;
      case 3: //NS:
        cit = sum(pow(var_obs - var_mod, 2));  //Nash-Sutcliffe efficiency
        jmen = sum(pow(var_obs - mean(var_obs), 2));
        ok = cit / jmen;  // minimization, i.e. not substracted from 1
        //ok = 1 - cit / jmen;
        break;
      case 4: //LNNS:  // does not work due to negative values of the var_obs
        cit = sum(pow(log(var_obs) - log(var_mod), 2));  //logarithmic Nash-Sutcliffe efficiency
        jmen = sum(pow(log(var_obs) - log(mean(var_obs)), 2));
        ok = cit / jmen;  // minimization, i.e. not substracted from 1
        //ok = 1 - cit / jmen;
        break;
      case 5: //MRE
          ok = mean((var_obs - var_mod) / var_obs); //mean relative error
        break;
      case 6: //tPI
        cit = sum(pow(var_obs.rows(1,(size(var_obs,0)-1)) - var_mod.rows(1,(size(var_mod,0)-1)), 2));
        jmen = sum(pow(var_mod.rows(1,(size(var_obs,0)-1)) - var_mod.rows(0,(size(var_obs,0)-2)), 2));
        ok = cit / jmen;  //transformed persistency index
        break;
      case 7: //PI
        cit = sum(pow(var_obs.rows(1,(size(var_obs,0)-1)) - var_mod.rows(1,(size(var_mod,0)-1)), 2));
        jmen = sum(pow(var_mod.rows(1,(size(var_obs,0)-1)) - var_mod.rows(0,(size(var_obs,0)-2)), 2));
        ok = cit / jmen;  // minimization, i.e. not substracted from 1
        //ok = 1-cit / jmen;  //persistency index
        break;
      case 8: //PID (tPI + MSE)
        cit = sum(pow(var_obs.rows(1,(size(var_obs,0)-1)) - var_mod.rows(1,(size(var_mod,0)-1)), 2));
        jmen = sum(pow(var_mod.rows(1,(size(var_obs,0)-1)) - var_mod.rows(0,(size(var_obs,0)-2)), 2));
        ok1 = cit / jmen;   //tPI
        ok2 = mean(pow((var_obs - var_mod), 2)); //MSE
        ok = ww*ok1 + (1-ww)*ok2;  //combined persistency index
        break;
      case 9: //dRMSE
        d_obs = var_obs.rows(0,(size(var_obs,0)-2)) - var_obs.rows(1,(size(var_obs,0)-1)) ;
        d_mod = var_mod.rows(0,(size(var_mod,0)-2)) - var_mod.rows(1,(size(var_mod,0)-1)) ;
        ok = pow(sum(pow(d_obs - d_mod, 4)), 0.25);   // root mean squared error in derivatives
        break;
      case 10: //PID1 (MAE + dRMSE)
        ok1 = mean(abs(var_obs - var_mod));   //MAE
        d_obs = var_obs.rows(0,(size(var_obs,0)-2)) - var_obs.rows(1,(size(var_obs,0)-1)) ;
        d_mod = var_mod.rows(0,(size(var_mod,0)-2)) - var_mod.rows(1,(size(var_mod,0)-1)) ;
        ok2 = pow(sum(pow(d_obs - d_mod, 4)), 0.25);   // dRMSE
        ok = ww*ok1 + (1-ww)*ok2;  //combined persistency index
        break;
//      case 11: //KGE
//        ok1 = mean(abs(var_obs - var_mod));   //MAE
//        d_obs = var_obs.rows(0,(size(var_obs,0)-2)) - var_obs.rows(1,(size(var_obs,0)-1)) ;
//        d_mod = var_mod.rows(0,(size(var_mod,0)-2)) - var_mod.rows(1,(size(var_mod,0)-1)) ;
//        ok2 = pow(sum(pow(d_obs - d_mod, 4)), 0.25);   // dRMSE
//        ok = ww*ok1 + (1-ww)*ok2;  //combined persistency index
//        break;
      default:
        break;
  }
  //out_fittness(ok,number_func_evaluation,"./data/outputs/ok_all");

  return(ok);

}


/**
 * Writing the fitness to the out file
 * \param value: fitness value, which will be written into the file
 * \param first_col: number which will be written in the first column
 * \param second_col: number which will be written in the second column
 * \param save_file: file for saving
 */
void mlp::out_fittness(double value, unsigned int first_col, unsigned int second_col, string save_file)
{
  ofstream out_stream(save_file.c_str(), ios::app);
    if (!out_stream) {
      cout << "\nIt is impossible to write to file  " << save_file;
      exit(EXIT_FAILURE);
    }
  out_stream << first_col << "\t" << second_col << "\t" << value <<"\n";
  out_stream.close();

}


/**
 * Writing the simulation outputs to the file (output variable, weigths, etc.)
 * \param data: matrix of simulated output, which will be written into the file
 * \param save_file: file for saving
 */
void mlp::out_outputs(mat data, string save_file)
{
  ofstream out_stream(save_file.c_str(), ios::app);
    if (!out_stream) {
      cout << "\nIt is impossible to write to file  " << save_file;
      exit(EXIT_FAILURE);
    }
  out_stream << data;
  out_stream.close();

}


/**
 * Run MLP model for validation, i.e. without optimization
 */
void mlp::runMLP()
{
  colvec help_simOutputs;
  help_simOutputs.set_size(nTargets);
  help_simOutputs.fill(1000);

  for (unsigned int k=0; k<nTargets; k++) {
    help_simOutputs(k) = calc_output(weights,trans_Inputs.row(k));
    simOutputs(k) = trans_to_orig(as_scalar(help_simOutputs(k)));
  }

  double ok_value = 0.0;
  ok_value = calc_fittnes(Pr_Number,obsOutputs,simOutputs);

  out_fittness(ok_value,1,Pr_Number,"./data/outputs/ok_val");
  out_outputs(simOutputs,"./data/outputs/best_best_simOutputs_val");
  out_outputs(weights,"./data/outputs/best_best_weights_val");

}
