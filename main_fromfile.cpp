#include <iostream>
#include <thread>
#include <vector>
#include <cstdlib>
#include <math.h>
#include <sstream>
#include <fstream>
#include <cassert>

using namespace std;

class TrainingData
{
public:
    TrainingData(const string filename);
    bool isEof(void)
    {
        return m_trainingDataFile.eof();
    }
    void getTopology(vector<int> &topology);

    // Returns the number of input values read from the file:
    unsigned getNextInputs(vector<double> &inputVals);
    unsigned getTargetOutputs(vector<double> &targetOutputVals);

private:
    ifstream m_trainingDataFile;
};

void TrainingData::getTopology(vector<int> &topology)
{
    string line;
    string label;

    getline(m_trainingDataFile, line);
    stringstream ss(line);
    ss >> label;
    if(this->isEof() || label.compare("topology:") != 0)
    {    //e.g., {3, 2, 1 }
        abort();
    }

    while(!ss.eof())
    {
        int n; // changed from unsigned
        ss >> n;
        topology.push_back(n);
    }
    return;
}

TrainingData::TrainingData(const string filename)
{
    m_trainingDataFile.open(filename.c_str());
}


unsigned TrainingData::getNextInputs(vector<double> &inputVals)
{
    inputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss >> label;
    if (label.compare("in:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            inputVals.push_back(oneValue);
        }
    }

    return inputVals.size();
}

unsigned TrainingData::getTargetOutputs(vector<double> &targetOutputVals)
{
    targetOutputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss>> label;
    if (label.compare("out:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            targetOutputVals.push_back(oneValue);
        }
    }

    return targetOutputVals.size();
}


struct Connect{

    double neu_weight;
    double neu_deltaweight;

};

class Neuron{

public:

    Neuron(int num_outputs,int m_index);
    void forward(vector<Neuron> &prev_layer);
    void assign_value(double x){neu_output_val = x; }
    double get_value(){ return neu_output_val;}
    void compute_delta(double x);
    void compute_hidden_delta(const vector<Neuron> &next);
    void update_weights(vector<Neuron> &prev);

private:
    double sigmoid_function(double);
    double sigmoid_function_derievative(double);
    double random_weight();
    double sum_derievative_next(const vector<Neuron> &next);
    double neu_output_val;
    vector<Connect> neu_output_weights;

    int index;
    double gradient;
    double learning_rate= 0.5;
    double momentum = 0.5;

};

typedef vector<Neuron> layer;

double Neuron::sigmoid_function(double x) { return 1 / (1 + exp(-x));}

double Neuron::sigmoid_function_derievative(double x) {return (sigmoid_function(x)*(1-sigmoid_function(x)));}


void Neuron::update_weights(vector<Neuron> &prev) {

    for (int i = 0; i < prev.size() ; ++i) {
        Neuron &n = prev[i];
        double old_delta_weight = n.neu_output_weights[index].neu_deltaweight;

        double new_delta_wight = learning_rate * n.get_value()*gradient + momentum*old_delta_weight;

        n.neu_output_weights[index].neu_deltaweight= new_delta_wight;
        n.neu_output_weights[index].neu_weight += new_delta_wight;
    }


}
void Neuron::forward(vector<Neuron> &prev_layer) {


    double temp = 0.0;

    for (int i = 0; i <prev_layer.size() ; ++i) {

        temp += prev_layer[i].get_value() * prev_layer[i].neu_output_weights[index].neu_weight;

    }

    neu_output_val = sigmoid_function(temp);


}


void Neuron::compute_delta(double x) {

    double temp = x - neu_output_val;
    gradient = temp * sigmoid_function_derievative(neu_output_val);


}

double Neuron::sum_derievative_next(const vector<Neuron> &next) {

    double tempsum = 0.0;

    for (int i = 0; i < next.size()-1; ++i) {

        tempsum += neu_output_weights[i].neu_weight * next[i].gradient;

    }

    return tempsum;
}

void Neuron::compute_hidden_delta(const vector<Neuron> &next) {

    double temp = sum_derievative_next(next);
    gradient = temp * sigmoid_function_derievative(neu_output_val);


}

Neuron::Neuron(int num_outputs,int m_index) {


    for (int i = 0; i <num_outputs ; ++i) {
        neu_output_weights.push_back(Connect());
        neu_output_weights.back().neu_weight = random_weight();


    }

    index = m_index;
}

double Neuron::random_weight() { return rand()/double(RAND_MAX);}


////////////////

class NeuralNet{

public:

    NeuralNet(const vector<int> &);
    void feedforward(const vector<double> &);
    void backProp(const vector<double> &);
    void getResult(vector<double> &) ;
    double get_error(void) const { return error; }


private:

    vector<vector<Neuron>> my_neuron;
    double error;


};

NeuralNet::NeuralNet(const vector<int> &input_values) {

    vector<Neuron> layer;
    int number_layers = input_values.size();
    int numOuts;
    for (int i = 0; i < number_layers; ++i) {

        my_neuron.push_back(layer);

        if (i == number_layers-1){

            numOuts = 1;

        }

        else {
            numOuts = input_values[i+1];
        }


        for (int j = 0; j <= input_values[i]; ++j) {

            my_neuron.back().emplace_back(Neuron(numOuts,i));

            cout<<"Made a Neuron!"<<endl;
        }

        my_neuron.back().back().assign_value(1.0);
    }

}

void NeuralNet::getResult(vector<double> &output_values) {

    output_values.clear();

    for (int i = 0; i < my_neuron.back().size() -1 ; ++i) {

        output_values.push_back(my_neuron.back()[i].get_value());

    }

}
void NeuralNet::feedforward(const vector<double> &input_values) {


    if(input_values.size()!=my_neuron[0].size()-1){

        cout<<"Error!!! Dimension mismatch!!!";
    }

    else

    {
        for (int i = 0; i < input_values.size() ; ++i) {
            my_neuron[0][i].assign_value(input_values[i]);
        }



        for (int i = 1; i <my_neuron.size() ; ++i) {

            layer &prev_layer = my_neuron[i-1];

            for (int j = 0; j <my_neuron[i].size()-1 ; ++j) {

                my_neuron[i][j].forward(prev_layer);

            }


            }

        }

    }

void NeuralNet::backProp(const vector<double> &actual_values) {

    layer &output_layer = my_neuron.back();

    error = 0.0;

    for (int i = 0; i < output_layer.size() - 1; ++i) {

        double temp = actual_values[i] - output_layer[i].get_value();
        error += temp * temp;
    }

    error /= output_layer.size() - 1;
    error = sqrt(error); //RMS error

    //Calculate Gradient for Output Layer

    for (int j = 0; j <output_layer.size()-1 ; ++j) {

        output_layer[j].compute_delta(actual_values[j]);
    }

    //Hidden Layer Gradients
    for (int k = my_neuron.size()-2; k >0 ; --k) {

        layer &hidden = my_neuron[k];
        layer &next = my_neuron[k+1];

        for (int i = 0; i < hidden.size() ; ++i) {
            hidden[i].compute_hidden_delta(next);
        }
    }

    //Update weights

    for (int l = my_neuron.size()-1; l >0 ; --l) {

        layer &present = my_neuron[l];
        layer &prev = my_neuron[l-1];

        for (int i = 0; i < present.size()-1 ; ++i) {

            present[i].update_weights(prev);
        }

    }
}

void showVectorVals(string label, vector<double> &v)
{
    cout << label << " ";
    for(unsigned i = 0; i < v.size(); ++i)
    {
        cout << v[i] << " ";
    }
    cout << endl;
}

int main()
{
    TrainingData trainData("trainingData.txt");
    vector<int> topology;
    trainData.getTopology(topology);
    NeuralNet myNet(topology);

    vector<double> inputVals, targetVals, resultVals;
    int trainingPass = 0;
    while(!trainData.isEof())
    {
        ++trainingPass;
        cout << endl << "Pass" << trainingPass;

        // Get new input data and feed it forward:
        if(trainData.getNextInputs(inputVals) != topology[0])
            break;
        showVectorVals(": Inputs :", inputVals);
        myNet.feedforward(inputVals);

        // Collect the net's actual results:
        myNet.getResult(resultVals);
        showVectorVals("Outputs:", resultVals);

        // Train the net what the outputs should have been:
        trainData.getTargetOutputs(targetVals);
        showVectorVals("Targets:", targetVals);
        assert(targetVals.size() == topology.back());

        myNet.backProp(targetVals);

        // Report how well the training is working, average over recnet
        cout << "Net recent average error: "
             << myNet.get_error() << endl;
    }

    cout << endl << "Done" << endl;

}