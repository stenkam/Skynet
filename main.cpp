#include <iostream>
#include <vector>
#include <cstdlib>
#include <assert.h>
#include <cmath>
using namespace std;

struct Conn{

    double weight;
    double deltaWeight;

};

class Neuron;

typedef vector<Neuron> Layer;

//Neuron:

class Neuron{

public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    void setOutputVal(double val){ m_output = val;}
    double getOutputVal() const {return m_output;}
    void feedForward(const Layer &prevLayer);
    void calcOutputGradients(double target);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);

private:
    static double eta;
    static double alpha;
    double transfer(double x);
    double transfer_D(double x);
    static double randomWeight(){ return rand()/double(RAND_MAX);}
    double sumDOW(const Layer &nextLayer) const;
    double m_output;
    vector<Conn> m_outputWeights;
    unsigned m_myIndex;
    double m_gradient;

};

double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;

void Neuron::updateInputWeights(Layer &prevLayer){

    for(unsigned n = 0; n < prevLayer.size(); n++){

        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

        double newDeltaWeight =

                eta *
                neuron.getOutputVal() *
                m_gradient +
                alpha *
                oldDeltaWeight;

        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;




    }

}
double Neuron::sumDOW(const Layer &nextLayer) const{

    double sum = 0.0;

    for (unsigned n = 0; n<nextLayer.size() - 1; n++){

        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;

    }

    return sum;

}

void Neuron::calcHiddenGradients(const Layer &nextLayer){

    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::transfer_D(m_output);

}

void Neuron::calcOutputGradients(double target){

    double delta = target - m_output;
    m_gradient = delta * Neuron::transfer_D(m_output);

}

double Neuron::transfer(double x){

    return tanh(x);

}

double  Neuron::transfer_D(double x){

    return 1.0-x*x;

}


Neuron::Neuron(unsigned numOutputs, unsigned myIndex){

    for(unsigned c = 0; c <= numOutputs;c++){

        m_outputWeights.push_back(Conn());
        m_outputWeights.back().weight = randomWeight();

    }

    m_myIndex=myIndex;
}
void Neuron::feedForward(const Layer &prevLayer){

    double sum=0.0;

    for (unsigned n = 0; n < prevLayer.size(); n++){

        sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].weight;

    }

    m_output = Neuron::transfer(sum);


}
//Network
class Net{

public:
    Net(const vector<unsigned> &top);
    void feedForward(const vector<double> &input);
    void backProp(const vector<double> &target);
    void getResults(vector<double> &result) const;


private:
    vector<Layer> m_layers;
    double m_error;
    double recent_AVG;
    double recent_ASF;

};
void Net::getResults(vector<double> &result) const{

    result.clear();

    for(unsigned n = 0; n < m_layers.back().size() - 1; n++){

        result.push_back(m_layers.back()[n].getOutputVal());

    }

}

void Net::backProp(const vector<double> &target){

    Layer & outputLayer = m_layers.back();

    m_error = 0.0;

    for (unsigned n = 0; n< outputLayer.size()-1; n++){

        double delta = target[n] - outputLayer[n].getOutputVal();
        m_error += delta * delta;

    }

    m_error /= outputLayer.size() - 1;
    m_error = sqrt(m_error);

    recent_AVG = (recent_AVG * recent_ASF + m_error)/(recent_ASF + 1.0);

    for (unsigned n = 0; n < outputLayer.size() - 1; n++){

        outputLayer[n].calcOutputGradients(target[n]);

    }

        for (unsigned n = outputLayer.size() - 2; n > 0; n--){

            Layer &hiddenLayer = m_layers[n];
            Layer &nextLayer = m_layers[n+1];

            for (unsigned m = 0; m < hiddenLayer.size(); n++){

                hiddenLayer[n].calcHiddenGradients(nextLayer);

            }
    }

    for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; layerNum--){

        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum - 1];

        for (unsigned n = 0; n< layer.size() - 1; n++){

            layer[n].updateInputWeights(prevLayer);

        }

    }


}


Net::Net(const vector<unsigned> &top){

    unsigned numLayers = top.size();

    for(unsigned layerNum = 0; layerNum < numLayers; layerNum++){

        m_layers.push_back(Layer());
        unsigned numOutputs = layerNum == top.size() - 1 ? 0 : top[layerNum + 1];

        for(unsigned neuronNum = 0;neuronNum <= top[layerNum];neuronNum++){

            m_layers.back().push_back(Neuron(numOutputs,neuronNum));
            //cout<<"Neuron: "<<neuronNum<<endl; //debug print

        }
    }

}

void Net::feedForward(const vector<double> &input){

    assert(input.size() == m_layers[0].size()-1);

    for (unsigned i = 0; i < input.size(); i++){

        m_layers[0][i].setOutputVal(input[i]);

    }


    for ( unsigned layerNum = 1; layerNum < m_layers.size(); layerNum++){

        Layer &prevLayer = m_layers[layerNum-1];

        for(unsigned n = 0; n<m_layers[layerNum].size() - 1; n++){

            m_layers[layerNum][n].feedForward(prevLayer);

        }

    }


}


int main()
{
    vector<unsigned> top;
    top.push_back(3);
    top.push_back(2);
    top.push_back(1);

    Net network(top);

    vector<double> input;
    network.feedForward(input);

    vector<double> target;
    network.backProp(target);

    vector<double> result;
    network.getResults(result);

    return 0;
}
