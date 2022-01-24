package nl.uva.aloha.examples;

import espam.utils.fileworker.ONNXFileWorker;
import nl.uva.aloha.Alterers.ONNXAlteration;

public class TestOnnxAlteration 
{
	public static void main(String args[]) 
	{
		
		String onnxpath = "oldfile.onnx";
		String onnxpathNew = "newfile.onnx";
		
		ONNXAlteration alterer = new ONNXAlteration(onnxpath);
		
		alterer.changeNeuronNumsOfLayer("ConvLayer3", 2);      //Name of a layer in oldfile.onnx
		alterer.changeNeuronNumsOfLayer("ConvLayer2", -3);     //Name of a layer in oldfile.onnx
		
		alterer.changeinputDimensionOfLayer("ConvLayer5", -3); //Name of a layer in oldfile.onnx
		alterer.changeinputDimensionOfLayer("dense1", 2);      //Name of a layer in oldfile.onnx
		
		ONNXFileWorker.writeModel(alterer.getOnnxModel(),onnxpathNew); //Save as a new file
	}
}
