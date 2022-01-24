package nl.uva.aloha;

import nl.uva.aloha.Alterers.ONNXAlteration;

public class ONNXAlter 
{
	
	public static void main(final String[] args) 
	{
		
		String onnxname = "sample.onnx";
		
		
		//Instance of ONNX Alter class is specific to the onnx. 
		//For multiple alterations, create a new instance per onnx file

		ONNXAlteration onnxAlterGt = new ONNXAlteration(onnxname);
		
		
		//Some examples of alteration functions
		
		
		onnxAlterGt.addBatchNormsAfterConv();
		
		//onnxAlterGt.removeReshapeLayer();
		//onnxAlterGt.correctpadding();
		//onnxAlterGt.resetInitializers();
		//onnxAlterGt.addBatchNormsAfterRelu();
		//onnxAlterGt.addDropOutLayersBetweenFC();
		//onnxAlterGt.addDropOutLayersBetweenFCRelu();
		//onnxAlterGt.addDropOutLayersBeforeFC();
		//onnxAlterGt.addChromosomeLevelSkipConnection();
		//onnxAlterGt.renameInputOutputLayerAfterOneRun();
		//onnxAlterGt.ReplaceFeatureExtractorFC();
		//onnxAlterGt.increaseOutputLayerby(1);
		
		
		//Till now all the changes are inside the ONNXAlteration class. 
		//Below function updates the actual file. 
		onnxAlterGt.updateONNXFile();
		
		
	}
	
}
