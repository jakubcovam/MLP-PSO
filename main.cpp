#include <iostream>
#include <fstream>
#include <string>

#include "mlp.h"
#include "pso.h"

using namespace std;

//class program
//{
//    public:
//        //enum { ok0_MSE, ok3_NS, ok7_PI, ok8_PID, ok10_PID1 } run_type_OK;
//        //enum { pso3_LinTimeVar, pso5_ChaoticRand, pso6_NonlinTimeConst, pso8_Adapt, pso9_APart } run_type_PSO;
//
//        program(); //!< The constructor
//        //{}
//        ~program();  //!< The destructor
//        //{}
//    protected:
//    private:
//};
//
//program::program()
//{
//  string run_type_OK = {"ok0_MSE", "ok3_NS", "ok7_PI", "ok8_PID", "ok10_PID1"};
//}
//
//program::~program()
//{
//}

int main()
{
    cout << "* Artificial neural network with multilayer perceptron *" << endl;
    cout << "********************************************************" << endl;

    vector<char> run_type_OK;
    //run_type_OK = {"ok0_MSE", "ok3_NS", "ok7_PI", "ok8_PID", "ok10_PID1"};
//    run_type_OK[1] = m;
    int a;

    for (unsigned int t_ok=0; t_ok<size(run_type_OK); t_ok++) {  // 5 PSO types
      //pso3_LinTimeVar;
      a=3;
  }

//!! RUN MLP MODELS WITH PSO OPTIMIZATION (i.e. calibration)
    cout << "\n--------------" ;
    cout << "\n*** 1. MLP ***" ;
  //!! 1. MLP
    //!! Read file with settings of the MLP
    mlp ann1("./settings/settings_ann_cal_1");
    //!! Read file with input data
    ann1.read_file("./data/inputs/input_data_cal");
    ann1.orig_to_trans();
    //!! Run the model with PSO optimization
    pso pso1("./settings/settings_pso_1",ann1);
    pso1.run_ensemble(ann1);

    cout << "\n--------------" ;
    cout << "\n*** 2. MLP ***" ;
  //!! 2. MLP
    //!! Read file with settings of the MLP
    mlp ann2("./settings/settings_ann_cal_2");
    //!! Read file with input data
    ann2.read_file("./data/inputs/input_data_cal");
    ann2.orig_to_trans();
    //!! Run the model with PSO optimization
    pso pso2("./settings/settings_pso_2",ann2);
    pso2.run_ensemble(ann2);

    cout << "\n--------------" ;
    cout << "\n*** 3. MLP ***" ;
  //!! 3. MLP
    //!! Read file with settings of the MLP
    mlp ann3("./settings/settings_ann_cal_3");
    //!! Read file with input data
    ann3.read_file("./data/inputs/input_data_cal");
    ann3.orig_to_trans();
    //!! Run the model with PSO optimization
    pso pso3("./settings/settings_pso_3",ann3);
    pso3.run_ensemble(ann3);

  cout << "\n--------------" ;
  cout << "\n*** 4. MLP ***" ;
  //!! 4. MLP
    //!! Read file with settings of the MLP
    mlp ann4("./settings/settings_ann_cal_4");
    //!! Read file with input data
    ann4.read_file("./data/inputs/input_data_cal");
    ann4.orig_to_trans();
    //!! Run the model with PSO optimization
    pso pso4("./settings/settings_pso_4",ann4);
    pso4.run_ensemble(ann4);

  cout << "\n--------------" ;
  cout << "\n*** Final MLP ***" ;
  //!! fin MLP
    //!! Read file with settings of the MLP
    mlp annF("./settings/settings_ann_cal_fin");  // number of inputs = number of previous mlp models + bias
    //!! Read file with input data
    annF.create_input("./data/inputs/input_data_cal","./data/outputs/best_best_simOutputs", "./data/inputs/input_data_cal_fin");  // inputs are the outputs from the previous 4 mlp models + last column with 1 (bias)
    annF.read_file("./data/inputs/input_data_cal_fin");  // read file with new dataset
    annF.orig_to_trans();
    //!! Run the model with PSO optimization
    pso psoF("./settings/settings_pso_fin",annF);
    psoF.run_ensemble(annF);



//!! RUN MLP MODELS WITHOUT OPTIMIZATION (i.e. validation)

 //!! Create input files (weigths, input data) from MLP with PSO to MLP without PSO

  //!! 1. MLP
    //!! Read file with settings of the MLP, with input data, with weights
    mlp annVal1("./settings/settings_ann_val_1");
    annVal1.read_file("./data/inputs/input_data_val");
    annVal1.orig_to_trans();
    //annVal1.create_weights("./data/outputs/best_best_weights", "./data/inputs/input_weights_val");
    annVal1.read_weights("./data/outputs/best_best_weights",1);
    //!! Run the model
    annVal1.runMLP();


  //!! 2. MLP
    //!! Read file with settings of the MLP, with input data, with weights
    mlp annVal2("./settings/settings_ann_val_2");
    annVal2.read_file("./data/inputs/input_data_val");
    annVal2.orig_to_trans();
    //annVal1.create_weights("./data/outputs/best_best_weights", "./data/inputs/input_weights_val");
    annVal2.read_weights("./data/outputs/best_best_weights",2);
    //!! Run the model
    annVal2.runMLP();


   //!! 3. MLP
    //!! Read file with settings of the MLP, with input data, with weights
    mlp annVal3("./settings/settings_ann_val_3");
    annVal3.read_file("./data/inputs/input_data_val");
    annVal3.orig_to_trans();
    //annVal1.create_weights("./data/outputs/best_best_weights", "./data/inputs/input_weights_val");
    annVal3.read_weights("./data/outputs/best_best_weights",3);
    //!! Run the model
    annVal3.runMLP();


  //!! 4. MLP
    //!! Read file with settings of the MLP, with input data, with weights
    mlp annVal4("./settings/settings_ann_val_4");
    annVal4.read_file("./data/inputs/input_data_val");
    annVal4.orig_to_trans();
    //annVal1.create_weights("./data/outputs/best_best_weights", "./data/inputs/input_weights_val");
    annVal4.read_weights("./data/outputs/best_best_weights",4);
    //!! Run the model
    annVal4.runMLP();


  //!! fin MLP
    //!! Read file with settings of the MLP, with input data, with weights
    mlp annValF("./settings/settings_ann_val_fin");
    annValF.create_input("./data/inputs/input_data_val", "./data/outputs/best_best_simOutputs_val", "./data/inputs/input_data_val_fin");  // inputs are the outputs from the previous 4 mlp models + last column with 1 (bias)
    annValF.read_file("./data/inputs/input_data_val_fin");
    annValF.orig_to_trans();
    annValF.read_weights("./data/outputs/best_best_weights",99);
    //!! Run the model
    annValF.runMLP();


/////******************************************************************************
////!! RUN MLP MODELS WITH PSO OPTIMIZATION (i.e. calibration)
//    cout << "\n--------------" ;
//    cout << "\n*** 1. MLP ***" ;
//  //!! 1. MLP
//    //!! Read file with settings of the MLP
//    mlp ann1("./ok0_MSE/pso3_LinTimeVar/settings/settings_ann_cal_1");
//    //!! Read file with input data
//    ann1.read_file("./ok0_MSE/pso3_LinTimeVar/data/inputs/input_data_cal");
//    ann1.orig_to_trans();
//    //!! Run the model with PSO optimization
//    pso pso1("./ok0_MSE/pso3_LinTimeVar/settings/settings_pso_1",ann1);
//    pso1.run_ensemble(ann1);
//
//    cout << "\n--------------" ;
//    cout << "\n*** 2. MLP ***" ;
//  //!! 2. MLP
//    //!! Read file with settings of the MLP
//    mlp ann2("./ok0_MSE/pso3_LinTimeVar/settings/settings_ann_cal_2");
//    //!! Read file with input data
//    ann2.read_file("./ok0_MSE/pso3_LinTimeVar/data/inputs/input_data_cal");
//    ann2.orig_to_trans();
//    //!! Run the model with PSO optimization
//    pso pso2("./ok0_MSE/pso3_LinTimeVar/settings/settings_pso_2",ann2);
//    pso2.run_ensemble(ann2);
//
//    cout << "\n--------------" ;
//    cout << "\n*** 3. MLP ***" ;
//  //!! 3. MLP
//    //!! Read file with settings of the MLP
//    mlp ann3("./ok0_MSE/pso3_LinTimeVar/settings/settings_ann_cal_3");
//    //!! Read file with input data
//    ann3.read_file("./ok0_MSE/pso3_LinTimeVar/data/inputs/input_data_cal");
//    ann3.orig_to_trans();
//    //!! Run the model with PSO optimization
//    pso pso3("./ok0_MSE/pso3_LinTimeVar/settings/settings_pso_3",ann3);
//    pso3.run_ensemble(ann3);
//
//  cout << "\n--------------" ;
//  cout << "\n*** 4. MLP ***" ;
//  //!! 4. MLP
//    //!! Read file with settings of the MLP
//    mlp ann4("./ok0_MSE/pso3_LinTimeVar/settings/settings_ann_cal_4");
//    //!! Read file with input data
//    ann4.read_file("./ok0_MSE/pso3_LinTimeVar/data/inputs/input_data_cal");
//    ann4.orig_to_trans();
//    //!! Run the model with PSO optimization
//    pso pso4("./ok0_MSE/pso3_LinTimeVar/settings/settings_pso_4",ann4);
//    pso4.run_ensemble(ann4);
//
//  cout << "\n--------------" ;
//  cout << "\n*** Final MLP ***" ;
//  //!! fin MLP
//    //!! Read file with settings of the MLP
//    mlp annF("./ok0_MSE/pso3_LinTimeVar/settings/settings_ann_cal_fin");  // number of inputs = number of previous mlp models + bias
//    //!! Read file with input data
//    annF.create_input("./ok0_MSE/pso3_LinTimeVar/data/inputs/input_data_cal","./ok0_MSE/pso3_LinTimeVar/data/outputs/best_best_simOutputs", "./ok0_MSE/pso3_LinTimeVar/data/inputs/input_data_cal_fin");  // inputs are the outputs from the previous 4 mlp models + last column with 1 (bias)
//    annF.read_file("./ok0_MSE/pso3_LinTimeVar/data/inputs/input_data_cal_fin");  // read file with new dataset
//    annF.orig_to_trans();
//    //!! Run the model with PSO optimization
//    pso psoF("./ok0_MSE/pso3_LinTimeVar/settings/settings_pso_fin",annF);
//    psoF.run_ensemble(annF);
//
//
//
////!! RUN MLP MODELS WITHOUT OPTIMIZATION (i.e. validation)
//
// //!! Create input files (weigths, input data) from MLP with PSO to MLP without PSO
//
//  //!! 1. MLP
//    //!! Read file with settings of the MLP, with input data, with weights
//    mlp annVal1("./ok0_MSE/pso3_LinTimeVar/settings/settings_ann_val_1");
//    annVal1.read_file("./ok0_MSE/pso3_LinTimeVar/data/inputs/input_data_val");
//    annVal1.orig_to_trans();
//    //annVal1.create_weights("./data/outputs/best_best_weights", "./data/inputs/input_weights_val");
//    annVal1.read_weights("./ok0_MSE/pso3_LinTimeVar/data/outputs/best_best_weights",1);
//    //!! Run the model
//    annVal1.runMLP();
//
//
//  //!! 2. MLP
//    //!! Read file with settings of the MLP, with input data, with weights
//    mlp annVal2("./ok0_MSE/pso3_LinTimeVar/settings/settings_ann_val_2");
//    annVal2.read_file("./ok0_MSE/pso3_LinTimeVar/data/inputs/input_data_val");
//    annVal2.orig_to_trans();
//    //annVal1.create_weights("./data/outputs/best_best_weights", "./data/inputs/input_weights_val");
//    annVal2.read_weights("./ok0_MSE/pso3_LinTimeVar/data/outputs/best_best_weights",2);
//    //!! Run the model
//    annVal2.runMLP();
//
//
//   //!! 3. MLP
//    //!! Read file with settings of the MLP, with input data, with weights
//    mlp annVal3("./ok0_MSE/pso3_LinTimeVar/settings/settings_ann_val_3");
//    annVal3.read_file("./ok0_MSE/pso3_LinTimeVar/data/inputs/input_data_val");
//    annVal3.orig_to_trans();
//    //annVal1.create_weights("./data/outputs/best_best_weights", "./data/inputs/input_weights_val");
//    annVal3.read_weights("./ok0_MSE/pso3_LinTimeVar/data/outputs/best_best_weights",3);
//    //!! Run the model
//    annVal3.runMLP();
//
//
//  //!! 4. MLP
//    //!! Read file with settings of the MLP, with input data, with weights
//    mlp annVal4("./ok0_MSE/pso3_LinTimeVar/settings/settings_ann_val_4");
//    annVal4.read_file("./ok0_MSE/pso3_LinTimeVar/data/inputs/input_data_val");
//    annVal4.orig_to_trans();
//    //annVal1.create_weights("./data/outputs/best_best_weights", "./data/inputs/input_weights_val");
//    annVal4.read_weights("./ok0_MSE/pso3_LinTimeVar/data/outputs/best_best_weights",4);
//    //!! Run the model
//    annVal4.runMLP();
//
//
//  //!! fin MLP
//    //!! Read file with settings of the MLP, with input data, with weights
//    mlp annValF("./ok0_MSE/pso3_LinTimeVar/settings/settings_ann_val_fin");
//    annValF.create_input("./ok0_MSE/pso3_LinTimeVar/data/inputs/input_data_val", "./ok0_MSE/pso3_LinTimeVar/data/outputs/best_best_simOutputs_val", "./ok0_MSE/pso3_LinTimeVar/data/inputs/input_data_val_fin");  // inputs are the outputs from the previous 4 mlp models + last column with 1 (bias)
//    annValF.read_file("./ok0_MSE/pso3_LinTimeVar/data/inputs/input_data_val_fin");
//    annValF.orig_to_trans();
//    annValF.read_weights("./ok0_MSE/pso3_LinTimeVar/data/outputs/best_best_weights",99);
//    //!! Run the model
//    annValF.runMLP();
/////******************************************************************************


    return 0;

}
