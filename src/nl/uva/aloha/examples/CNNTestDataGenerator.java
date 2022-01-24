package nl.uva.aloha.examples;

import java.util.Vector;

import espam.datamodel.graph.cnn.BoundaryMode;
import espam.datamodel.graph.cnn.Network;
import espam.datamodel.graph.cnn.connections.ConnectionType;
import espam.datamodel.graph.cnn.neurons.cnn.Convolution;
import espam.datamodel.graph.cnn.neurons.cnn.Pooling;
import espam.datamodel.graph.cnn.neurons.generic.GenericNeuron;
import espam.datamodel.graph.cnn.neurons.neurontypes.DataType;
import espam.datamodel.graph.cnn.neurons.neurontypes.NonLinearType;
import espam.datamodel.graph.cnn.neurons.neurontypes.PoolingType;
import espam.datamodel.graph.cnn.neurons.simple.Data;
import espam.datamodel.graph.cnn.neurons.simple.DenseBlock;
import espam.datamodel.graph.cnn.neurons.simple.NonLinear;
import espam.datamodel.graph.cnn.neurons.transformation.Concat;
import 	espam.datamodel.graph.csdf.datasctructures.Tensor;;

/**
 * Class to generate test data for testing internal CNN model
 */
public class CNNTestDataGenerator {

	public static void main(String[] args) {
		generateTestNetworks();
	}
	
    /**generate all available test neural networks*/
    public static Vector<Network> generateTestNetworks() {
        Vector<Network> networks = new Vector<Network>();
        networks.add(createLenetSimple("simpleLeNet"));
        networks.add(createLenetWithGenericNeurons("lenetWithGenericNeurons"));
        networks.add(createLenetWithGenericNeuronsAndConcat("lenetWithGenericNeuronsAndConcat"));
        return networks;
    } 

    //////////////////////
    ////Test Networks////

    public static Network createLenetSimple(String name){
     System.out.println(name+" CNN model creation...");
     Network lenet = lenetModel();
     lenet.setName(name);
     System.out.println(name+" CNN model created.");
     return lenet;
    }


    public static Network createLenetWithGenericNeurons(String name){
     System.out.println(name+" CNN model creation...");
     Network lenetComplex= lenetWithComplexNeurons(name);
     System.out.println(name+" CNN model created.");
     return lenetComplex;
    }

     public static Network createLenetWithGenericNeuronsAndConcat(String name){
     System.out.println(name+" CNN model creation...");
     Network lenetComplex= LeNetWithGenericAndConcat(name);
     System.out.println(name+" CNN model created.");
     return lenetComplex;
    }

    public static Network createLenetAsBlackBox(String name) {
     System.out.println(name+" CNN model creation...");
     Network lenetComplex= lenetAsBlackBox(name);
     System.out.println(name+" CNN model created.");
     return lenetComplex;
    }

    ///////////////////////////
    ////Test Networks parts////

    public static Network createLenetConvBlock(String name,int neuronsNum)
    {  System.out.println(name+" CNN model creation...");
       Network net = LeNetConvSubNetworkWithDataL(name,neuronsNum);
       net.setName(name);
       System.out.println(name+" CNN model created.");
       return net;
    }

    public static Network createLenetPartWithCustomConnection(String name)
    {  System.out.println(name+" CNN model creation...");
       Network net = buildTestSubNetworkWithCustomConn(name);
       System.out.println(name+" CNN model created.");
       return net;
    }
    
    public static Network lenetAsBlackBox(String name) {
          Network lenetAsBlackBox = new Network(name);
          Data input = new Data(DataType.INPUT);
          lenetAsBlackBox.addLayer("input",input,1);

          Network lenet = CNNTestDataGenerator.lenetModelInternalStructure();
          GenericNeuron neuronWithLenet = new GenericNeuron("neuronWithLenet",lenet);
          lenetAsBlackBox.stackLayer("layer",neuronWithLenet,1,ConnectionType.ONETOALL);

          Data output = new Data(DataType.OUTPUT);
          lenetAsBlackBox.stackLayer("output",output,1, ConnectionType.ALLTOALL);


          lenetAsBlackBox.setInputLayer(lenetAsBlackBox.getLayers().firstElement());
          lenetAsBlackBox.setOutputLayer(lenetAsBlackBox.getLayers().lastElement());
          
          return lenetAsBlackBox;
        
    }

    public static Network lenetModel() {
         /**
          * Google LeNet model, described in GLeNet.java
        */
       Network lenet = new Network("lenet");

        Data inputData = new Data(DataType.INPUT);
        lenet.addLayer("dataLayer",inputData,1);
         /**
          * First convolutional layer:
          * filter = 5x5,stride = 1, maps = 6@28x28
         *  ConvLayer1
         * {
         *     filtersTotal: 6,
         *     kernelSize: 5,
         *     stride: 1,
         *     maps: 6,
         *     input: 'data',
         *     output: 'PoolingLayer1'
         * }
          */

         Convolution conv1 = new Convolution(5, BoundaryMode.VALID,1);
         lenet.stackLayer("ConvLayer1",conv1,6,ConnectionType.ONETOALL);

         /**
          * first pooling layer
          * pooling [2x2],stride=2, fn = avg
         * PoolingLayer1
         * {
         *     filtersTotal: 6,
         *     kernelSize: 2,
         *     stride: 2,
         *     input: 'ConvLayer1',
         *     output: 'ReLuLayer1'
         * }
          *
          *
          *
          */
         Pooling pool1 = new Pooling(PoolingType.MAXPOOL,2);
         lenet.stackLayer("PoolingLayer1",pool1,6,ConnectionType.ONETOONE);

         /**
         * ReLuLayer1
         * {
         *     inputsTotal: 6@14x14,
         *     input: 'PoolingLayer1',
         *     output: 'ConvLayer2'
         * }
          */
         /**
          * One node of first ReLu layer = Node for operating over the whole input data chunk
          * of one feature map
            */
         NonLinear relu1 = new NonLinear(NonLinearType.ReLU);
         lenet.stackLayer("ReLuLayer1", relu1,6,ConnectionType.ONETOONE);

         /**
          * filter = 5x5x6, maps = 16
         * ConvLayer2
         * {
         *     filtersTotal: 16,
         *     kernelSize: 5,
         *     stride: 1,
         *     input: 'ReluLayer1',
         *     output: 'PoolingLayer2'
         * }
          */

         /**
          * Google LeNet connection Matrix
          */
         int [][] connectionMatrix =
         {{1,0,0,0,1,1,1,0,0,1,1,1,1,0,1,1},
          {1,1,0,0,0,1,1,1,0,0,1,1,1,1,0,1},
          {1,1,1,0,0,0,1,1,1,0,0,1,0,1,1,1},
          {0,1,1,1,0,0,1,1,1,1,0,0,1,0,1,1},
          {0,0,1,1,1,0,0,1,1,1,1,0,1,1,0,1},
          {0,0,0,1,1,1,0,0,1,1,1,1,0,1,1,1}};

         boolean[][] booleanConnection =
          {{true,false,false,false,true,true,true,false,false,true,true,true,true,false,true,true},
          {true,true,false,false,false,true,true,true,false,false,true,true,true,true,false,true},
          {true,true,true,false,false,false,true,true,true,false,false,true,false,true,true,true},
          {false,true,true,true,false,false,true,true,true,true,false,false,true,false,true,true},
          {false,false,true,true,true,false,false,true,true,true,true,false,true,true,false,true},
          {false,false,false,true,true,true,false,false,true,true,true,true,false,true,true,true}};

          /**
          * One node of the second convolutional layer
          */
         Convolution conv2 = new Convolution(5, BoundaryMode.VALID,1);
         //lenet.stackLayer("ConvLayer2",conv2,16,booleanConnection); DOLLY
         lenet.stackLayer("ConvLayer2",conv2,16,ConnectionType.ONETOALL);


         NonLinear relu2 = new NonLinear(NonLinearType.ReLU);
         lenet.stackLayer("ReLuLayer2", relu2,16,ConnectionType.ONETOONE);
         
         Convolution conv5 = new Convolution(5, BoundaryMode.VALID,1);
         //lenet.stackLayer("ConvLayer2",conv2,16,booleanConnection); DOLLY
         lenet.stackLayer("ConvLayer5",conv5,16,ConnectionType.ONETOALL);


         NonLinear relu5 = new NonLinear(NonLinearType.ReLU);
         lenet.stackLayer("ReLuLayer5", relu5,16,ConnectionType.ONETOONE);
         
         /**pooling [2x2],stride=2, fn = avg
         * PoolingLayer2
         * {
         *     filtersTotal: 16,
         *     kernelSize: 2,
         *     stride: 2,
         *     input: 'ConvLayer2',
         *     output: 'ReLuLayer2'
         * }*/

         Pooling pool2 = new Pooling(PoolingType.MAXPOOL,1);
         lenet.stackLayer("PoolingLayer2",pool2,16,ConnectionType.ONETOONE);

         
         
        // ADDED BY DOLLY - testing onnx
         Convolution conv3 = new Convolution(5, BoundaryMode.VALID,1);
         //lenet.stackLayer("ConvLayer2",conv2,16,booleanConnection); DOLLY
         lenet.stackLayer("ConvLayer3",conv3,32,ConnectionType.ONETOALL);

         NonLinear relu3 = new NonLinear(NonLinearType.ReLU);
         lenet.stackLayer("ReLuLayer3", relu3,32,ConnectionType.ONETOONE);
//         
//         Convolution conv4 = new Convolution(5, BoundaryMode.VALID,1);
//         //lenet.stackLayer("ConvLayer2",conv2,16,booleanConnection); DOLLY
//         lenet.stackLayer("ConvLayer4",conv4,32,ConnectionType.ONETOALL);
//
//         NonLinear relu4 = new NonLinear(NonLinearType.ReLU);
//         lenet.stackLayer("ReLuLayer4", relu4,32,ConnectionType.ONETOONE);
         
         Pooling pool3 = new Pooling(PoolingType.MAXPOOL,1);
         lenet.stackLayer("PoolingLayer3",pool3,32,ConnectionType.ONETOONE);

        
         

         /** DenseLayer1
         * {
         *     inputsTotal: 16@5x5,
         *     neurons: 120,
         *     input: 'ReLuLayer2',
         *     output: 'DenseLayer2'
         * }*/

        /**
         * Dense Layer is considered to be one block element
         */
         DenseBlock dense1 = new DenseBlock(256);
         lenet.stackLayer("dense1", dense1,1,ConnectionType.ALLTOALL);

         /**
         * DenseLayer2
         * {
         *     inputsTotal: 1@120,
         *     neurons: 84,
         *     input: 'DenseLayer1',
         *     output: 'SoftmaxLayer1'
         * }*/

            /**
         * Dense Layer is considered to be one block element
         */
         DenseBlock dense2 = new DenseBlock(128);
         lenet.stackLayer("dense2", dense2,1,ConnectionType.ALLTOALL);

         DenseBlock dense3 = new DenseBlock(10);
         lenet.stackLayer("dense3", dense3,1,ConnectionType.ALLTOALL);
         
         /**
         * SoftMaxLayer1
         * {
         *     inputsTotal: 1@84,
         *     neurons: 10,
         *     input: 'DenseLayer2',
         *     output: 'output'
         * }
         */

        // DenseBlock softmax = new DenseBlock(NonLinearType.SOFTMAX,10);
         NonLinear softmax = new NonLinear( NonLinearType.SOFTMAX);
         lenet.stackLayer("softmax", softmax,10,ConnectionType.ALLTOALL);

         /**
         * Output layer
         * {
         *     inputsTotal: 1@10,
         * }
         */
         Data output = new Data(DataType.OUTPUT);
         lenet.stackLayer("output", output,1,ConnectionType.ALLTOALL);

        /**
         * set input and output layer Ids
         */

        lenet.setInputLayer(lenet.getLayers().firstElement());
         lenet.setOutputLayer(lenet.getLayers().lastElement());

        return lenet;
    }

    /**
     * Return lenet without input and output data layers
     * @return lenet without input and output data layers
     */
    public static Network lenetModelInternalStructure() {
         /**
          * Google LeNet model, described in GLeNet.java
        */
         Network lenet = new Network("lenetModelInternalStructure");

         Convolution conv1 = new Convolution(5, BoundaryMode.VALID,1);
         lenet.addLayer("ConvLayer1",conv1,6);

         Pooling pool1 = new Pooling(PoolingType.MAXPOOL,2);
         lenet.stackLayer("PoolingLayer1",pool1,6,ConnectionType.ONETOONE);

         NonLinear relu1 = new NonLinear(NonLinearType.ReLU);
         lenet.stackLayer("ReLuLayer1", relu1,6,ConnectionType.ONETOONE);

         /**
          * Google LeNet connection Matrix
          */

         boolean[][] booleanConnection =
          {{true,false,false,false,true,true,true,false,false,true,true,true,true,false,true,true},
          {true,true,false,false,false,true,true,true,false,false,true,true,true,true,false,true},
          {true,true,true,false,false,false,true,true,true,false,false,true,false,true,true,true},
          {false,true,true,true,false,false,true,true,true,true,false,false,true,false,true,true},
          {false,false,true,true,true,false,false,true,true,true,true,false,true,true,false,true},
          {false,false,false,true,true,true,false,false,true,true,true,true,false,true,true,true}};


         Convolution conv2 = new Convolution(5, BoundaryMode.VALID,1);
         lenet.stackLayer("ConvLayer2",conv2,16,booleanConnection);

         Pooling pool2 = new Pooling(PoolingType.MAXPOOL,2);
         lenet.stackLayer("PoolingLayer2",pool2,16,ConnectionType.ONETOONE);

         NonLinear relu2 = new NonLinear(NonLinearType.ReLU);
         lenet.stackLayer("ReLuLayer2", relu2,16,ConnectionType.ONETOONE);

         DenseBlock dense1 = new DenseBlock(120);
         lenet.stackLayer("dense1", dense1,1,ConnectionType.ALLTOALL);

         DenseBlock dense2 = new DenseBlock(84);
         lenet.stackLayer("dense2", dense2,1,ConnectionType.ALLTOALL);

         DenseBlock softmax = new DenseBlock(NonLinearType.SOFTMAX,10);
         lenet.stackLayer("softmax", softmax,1,ConnectionType.ALLTOALL);

         lenet.setInputLayer(lenet.getLayers().firstElement());
         lenet.setOutputLayer(lenet.getLayers().lastElement());

        return lenet;
    }

    /**
     * Create LeNet with generic blocks for test
     * @param name name of the subnetwork
     * @return dummy dense subnetwork for test
     */

    public static Network LeNetWithGenericAndConcat(String name) {
    Network complexNet = new Network(name);

    Data inputData = new Data(DataType.INPUT);
    complexNet.addLayer("dataLayer", inputData, 1);

    /**
     * First Conv subnetwork parameters
     */
    int totalConvs = 6;
    int genericNeuronsNum = 2;
    int neuronsPerBlock = totalConvs/genericNeuronsNum;
    Network convSubnetwork = convSubNetwork("convSubnetwork1",neuronsPerBlock);
    GenericNeuron genericNeuron = new GenericNeuron("convBlockNeuron1", convSubnetwork);
    /**
     * Block 1
     * Convolutional subnetwork
     */
    complexNet.stackLayer("layer1", genericNeuron, genericNeuronsNum, ConnectionType.ONETOALL);

    /**
     * Concat results of the first computational block
     */
    Concat concat = new Concat();//complexNet.getLayers().lastElement().getNeuronsNum());
    complexNet.stackLayer("concat",concat,1,ConnectionType.ALLTOONE);

    /**
     * Second Conv subnetwork parameters
     */
    totalConvs = 16;
    genericNeuronsNum = 4;
    neuronsPerBlock = totalConvs/genericNeuronsNum;
    Network convSubnetwork2 = convSubNetwork("convSubnetwork2",neuronsPerBlock);
    GenericNeuron genericNeuron2 = new GenericNeuron("convBlockNeuron2", convSubnetwork2);
    /**
     * Block 2
     * Convolutional subnetwork
     */
    complexNet.stackLayer("layer2", genericNeuron2, genericNeuronsNum, ConnectionType.ALLTOALL);

    /**
     * Concat results of the second computational block
     */
    Concat concat2 = new Concat();//complexNet.getLayers().lastElement().getNeuronsNum());
    complexNet.stackLayer("concat2",concat2,1,ConnectionType.ALLTOONE);

    /**
     * Block3
     * Dense subnetwork
     */
    Network denseSubnetwork = denseSubNetwork("denseSubNetwork");
    GenericNeuron denseGenericNeuron = new GenericNeuron("denseBlockNeuron", denseSubnetwork);
    complexNet.stackLayer("layer3", denseGenericNeuron, 1,ConnectionType.ALLTOALL);

    /**
     * Output layer
     * {
     *     inputsTotal: 1@10
     * }
     */
    Data output = new Data(DataType.OUTPUT);
    complexNet.stackLayer("Output", output,1, ConnectionType.ALLTOALL);

    /**
    * Set input and output layers
    */
    complexNet.setInputLayer(complexNet.getLayers().firstElement());
    complexNet.setOutputLayer(complexNet.getLayers().lastElement());

    return complexNet;
}

    /**
     * Build leNet model with complex (generic) neurons
     * @param name name of the model
     * @return leNet model with complex (generic) neurons
     */
    public static Network lenetWithComplexNeurons(String name) {
    Network complexNet = new Network(name);

    Data inputData = new Data(DataType.INPUT);
    complexNet.addLayer("dataL", inputData, 1);

    /**
     * First Conv subnetwork generic neuron
     */
    int totalConvs = 6;
    int genericNeuronsNum = 2;
    int neuronsPerBlock = totalConvs/genericNeuronsNum;
    Network convSubnetwork =  convSubNetwork("convSubnetwork1",neuronsPerBlock);
    GenericNeuron genericNeuron = new GenericNeuron("convBlockNeuron1", convSubnetwork);
    /**
     * Block 1
     * Convolutional subnetwork
     */
    complexNet.stackLayer("layer1", genericNeuron, genericNeuronsNum,ConnectionType.ONETOALL);

    /**
     * Second Conv subnetwork generic neuron
     */
    totalConvs = 16;
    genericNeuronsNum = 4;
    neuronsPerBlock = totalConvs/genericNeuronsNum;
    Network convSubnetwork2 = convSubNetwork("convSubnetwork2",neuronsPerBlock);
    GenericNeuron genericNeuron2 = new GenericNeuron("convBlockNeuron2", convSubnetwork2);
    /**
     * Block 2
     * Convolutional subnetwork
     */
    complexNet.stackLayer("layer2", genericNeuron2, genericNeuronsNum,ConnectionType.ALLTOALL);

    /**
     * Block3
     * Dense subnetwork
     */
    Network denseSubnetwork = denseSubNetwork("denseSubNetwork");
    GenericNeuron denseGenericNeuron = new GenericNeuron("denseBlockNeuron", denseSubnetwork);
    complexNet.stackLayer("layer3", denseGenericNeuron, 1,ConnectionType.ALLTOALL);

    /**
     * Output layer
     * {
     *     inputsTotal: 1@10
     * }
     */
    Data output = new Data(DataType.OUTPUT);
    complexNet.stackLayer("Output", output,1,ConnectionType.ALLTOALL);

    /**
    * Set input and output layers
    */
    complexNet.setInputLayer(complexNet.getLayers().firstElement());
    complexNet.setOutputLayer(complexNet.getLayers().lastElement());

    return complexNet;
    }

    /**
     * Create part LeNet Convolutional Subnetwork (first three blocks) with generic blocks for test
     * @param name name of the subnetwork
     * @return dummy dense subnetwork for test
     */
    public static Network LeNetConvSubNetworkWithDataL(String name,int neuronsNum) {
    Network block = new Network(name);
    Data inputData = new Data(DataType.INPUT);
    block.addLayer("dataL", inputData, 1);

     /**
          * First convolutional layer:
          * filter = 5x5,stride = 1, maps = 6@28x28
         *  ConvLayer1
         * {
         *     filtersTotal: 6,
         *     kernelSize: 5,
         *     stride: 1,
         *     maps: 6,
         *     input: 'data',
         *     output: 'PoolingLayer1'
         * }
          */

         Convolution conv1 = new Convolution(5, BoundaryMode.VALID,1);
         block.stackLayer("ConvLayer1",conv1,neuronsNum,ConnectionType.ONETOALL);

         /**
          * first pooling layer
          * pooling [2x2],stride=2, fn = avg
         * PoolingLayer1
         * {
         *     filtersTotal: 6,
         *     kernelSize: 2,
         *     stride: 2,
         *     input: 'ConvLayer1',
         *     output: 'ReLuLayer1'
         * }
         */
         Pooling pool1 = new Pooling(PoolingType.MAXPOOL,2);
         block.stackLayer("PoolingLayer1",pool1,neuronsNum,ConnectionType.ONETOONE);

         block.setInputLayer(block.getLayers().firstElement());
         block.setOutputLayer(block.getLayers().lastElement());

         return block;
    }

    /**
     * Create part LeNet Convolutional Subnetwork (first four blocks) with generic blocks
     * and custom connection for test
     * @param name name of the subnetwork
     * @return dummy dense subnetwork for test
     */
    public static Network buildTestSubNetworkWithCustomConn(String name) {

        Network lenet = new Network(name);

        Data inputData = new Data(DataType.INPUT);
        lenet.addLayer("dataLayer",inputData,1);

        Convolution conv1 = new Convolution(5, BoundaryMode.VALID,1);
        lenet.stackLayer("ConvLayer1",conv1,6,ConnectionType.ONETOALL);

        Pooling pool1 = new Pooling(PoolingType.MAXPOOL,2);
        lenet.stackLayer("PoolingLayer1",pool1,6,ConnectionType.ONETOONE);

        NonLinear relu1 = new NonLinear(NonLinearType.ReLU);
        lenet.stackLayer("ReLuLayer1", relu1,6,ConnectionType.ONETOONE);

         /**
          * Google LeNet connection Matrix
          */
         int [][] connectionMatrix =
         {{1,0,0,0,1,1,1,0,0,1,1,1,1,0,1,1},
          {1,1,0,0,0,1,1,1,0,0,1,1,1,1,0,1},
          {1,1,1,0,0,0,1,1,1,0,0,1,0,1,1,1},
          {0,1,1,1,0,0,1,1,1,1,0,0,1,0,1,1},
          {0,0,1,1,1,0,0,1,1,1,1,0,1,1,0,1},
          {0,0,0,1,1,1,0,0,1,1,1,1,0,1,1,1}};

         boolean[][] booleanConnection =
          {{true,false,false,false,true,true,true,false,false,true,true,true,true,false,true,true},
          {true,true,false,false,false,true,true,true,false,false,true,true,true,true,false,true},
          {true,true,true,false,false,false,true,true,true,false,false,true,false,true,true,true},
          {false,true,true,true,false,false,true,true,true,true,false,false,true,false,true,true},
          {false,false,true,true,true,false,false,true,true,true,true,false,true,true,false,true},
          {false,false,false,true,true,true,false,false,true,true,true,true,false,true,true,true}};

          /**
          * One node of the second convolutional layer
          */
         Convolution conv2 = new Convolution(5, BoundaryMode.VALID,1);
         lenet.stackLayer("ConvLayer2",conv2,16,booleanConnection);
        /**
         * Set input and output layers
         */
        lenet.setInputLayer(lenet.getLayers().firstElement());
        lenet.setOutputLayer(lenet.getLayers().lastElement());

         return lenet;
    }

    /**
     * Create all Convolutional part of lenet
     * @param name name of the subnetwork
     * @return all Convolutional part of lenet for test
     */
    public static Network allConvulutionalPart(String name) {

        Network lenet = new Network(name);

        Data inputData = new Data(DataType.INPUT);
        lenet.addLayer("dataLayer",inputData,1);

        Convolution conv1 = new Convolution(5, BoundaryMode.VALID,1);
        lenet.stackLayer("ConvLayer1",conv1,6,ConnectionType.ONETOALL);

        Pooling pool1 = new Pooling(PoolingType.MAXPOOL,2);
        lenet.stackLayer("PoolingLayer1",pool1,6,ConnectionType.ONETOONE);

        NonLinear relu1 = new NonLinear(NonLinearType.ReLU);
        lenet.stackLayer("ReLuLayer1", relu1,6,ConnectionType.ONETOONE);

         /**
          * Google LeNet connection Matrix
          */
         int [][] connectionMatrix =
         {{1,0,0,0,1,1,1,0,0,1,1,1,1,0,1,1},
          {1,1,0,0,0,1,1,1,0,0,1,1,1,1,0,1},
          {1,1,1,0,0,0,1,1,1,0,0,1,0,1,1,1},
          {0,1,1,1,0,0,1,1,1,1,0,0,1,0,1,1},
          {0,0,1,1,1,0,0,1,1,1,1,0,1,1,0,1},
          {0,0,0,1,1,1,0,0,1,1,1,1,0,1,1,1}};

         boolean[][] booleanConnection =
          {{true,false,false,false,true,true,true,false,false,true,true,true,true,false,true,true},
          {true,true,false,false,false,true,true,true,false,false,true,true,true,true,false,true},
          {true,true,true,false,false,false,true,true,true,false,false,true,false,true,true,true},
          {false,true,true,true,false,false,true,true,true,true,false,false,true,false,true,true},
          {false,false,true,true,true,false,false,true,true,true,true,false,true,true,false,true},
          {false,false,false,true,true,true,false,false,true,true,true,true,false,true,true,true}};

         Convolution conv2 = new Convolution(5, BoundaryMode.VALID,1);
         lenet.stackLayer("ConvLayer2",conv2,16,booleanConnection);

         Pooling pool2 = new Pooling(PoolingType.MAXPOOL,2);
         lenet.stackLayer("PoolingLayer2",pool2,16,ConnectionType.ONETOONE);

         NonLinear relu2 = new NonLinear(NonLinearType.ReLU);
         lenet.stackLayer("ReLuLayer2", relu2,16,ConnectionType.ONETOONE);

         /**
         * Set input and output layers
         */
        lenet.setInputLayer(lenet.getLayers().firstElement());
        lenet.setOutputLayer(lenet.getLayers().lastElement());

         return lenet;
    }

    /**
     * Create part LeNet Convolutional Subnetwork (convolution->pooling->nonlinear)
     * @param neuronsNum number of neurons of ech sub-layer(convolution, pooling and nonlinear)
     * @param name name of the subnetwork
     * @return dummy convolutional subnetwork for test
     */

    public static Network convSubNetwork(String name,int neuronsNum) {

        Network block = new Network(name);

         Convolution conv1 = new Convolution(5, BoundaryMode.VALID,1);
         block.addLayer("ConvLayer1",conv1,neuronsNum);

         Pooling pool1 = new Pooling(PoolingType.MAXPOOL,2);
         block.stackLayer("PoolingLayer1",pool1,neuronsNum,ConnectionType.ONETOONE);

         /**
         * Set input and output layers
         */
        block.setInputLayer(block.getLayers().firstElement());
        block.setOutputLayer(block.getLayers().lastElement());

         return block;
    }

     /**
     * Create dummy dense subnetwork for test
     * @param name name of the subnetwork
     * @return dummy dense subnetwork for test
     */
    public static Network denseSubNetwork(String name) {

        Network block = new Network(name);

        /**
         * Dense Layer is considered to be one block element
         */
         DenseBlock dense1 = new DenseBlock(120);
         block.addLayer("dense1", dense1,1);

         DenseBlock dense2 = new DenseBlock(84);
         block.stackLayer("dense2", dense2,1,ConnectionType.ALLTOALL);

         DenseBlock softmax = new DenseBlock(NonLinearType.SOFTMAX,10);
         block.stackLayer("softmax", softmax,1,ConnectionType.ALLTOALL);

         /**
         * Set input and output layers
         */
         block.setInputLayer(block.getLayers().firstElement());
         block.setOutputLayer(block.getLayers().lastElement());

         return block;
    }

    /**
     * Create first block of ResidualNet
     * @param inputDataFormat input data format
     * @return first block of ResidualNet
     */
    public static Network resNetBlock1(Tensor inputDataFormat) {
        Network block = new Network("resNetBlock1");
        /** Input Layer*/
        Data input = new Data(DataType.INPUT);
        block.addLayer("dataL", input, 1);

        /**
         * First convolutional layer:
         * filter = 3x3,stride = 1, maps = 32@149x149
         */
        Convolution conv1 = new Convolution(3, BoundaryMode.VALID,1);
        block.stackLayer("ConvLayer1",conv1,1, ConnectionType.ONETOALL);
        /**
         * Second convolutional layer:
         * filter = 3x3,stride = 1, maps = 32@147x147
         */
        Convolution conv2 = new Convolution(3, BoundaryMode.VALID,1);
        block.stackLayer("ConvLayer2",conv2,1, ConnectionType.ALLTOALL);
        /**
         * Third convolutional layer:
         * filter = 3x3,stride = 1, maps = 64@147x147
         */
        Convolution conv3 = new Convolution(3, BoundaryMode.SAME,1);
        block.stackLayer("ConvLayer3",conv3,2, ConnectionType.ALLTOALL);
        /**
         * Fourth convolutional layer - parallel block to PoolingLayer1
         * filter = 3x3,stride = 2, maps = 96@73x73
         */
        Convolution conv4 = new Convolution(3, BoundaryMode.VALID,2);
        block.stackLayer("ConvLayer4",conv4,3,ConnectionType.ALLTOALL);

        /**
         * First pooling layer - parallel block to ConvLayer4
         * filter = 3x3,stride = 2, maps = 96@73x73
         */
        Pooling pool1 = new Pooling(PoolingType.MAXPOOL,2);
        block.addLayer("MaxPool1",pool1,2);

        /**
         * add parallel connection
         */
        block.addConnection(block.getLayer("ConvLayer3"),block.getLayer("MaxPool1"),ConnectionType.ONETOONE);

        /**
         * Concat Layer, inputs: ConvLayer4, MaxPool1
         */
        Concat concat1 = new Concat();
        block.addLayer("concat1",concat1,1);

        block.addConnection(block.getLayer("ConvLayer4"),block.getLayer("concat1"),ConnectionType.ALLTOONE);
        block.addConnection(block.getLayer("MaxPool1"),block.getLayer("concat1"),ConnectionType.ALLTOONE);

        /**
         * Output layer
         */
        Data output = new Data(DataType.OUTPUT);
        block.stackLayer("Output",output,1, ConnectionType.ALLTOONE);

        /**
         * set input and output layers
         */
        block.setInputLayer(block.getLayers().firstElement());
        block.setOutputLayer(block.getLayers().lastElement());

        return block;
    }
}
