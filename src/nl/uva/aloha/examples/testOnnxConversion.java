package nl.uva.aloha.examples;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Iterator;
import java.util.List;

import com.google.protobuf.ByteString;

import  onnx.ONNX.GraphProto;
import  onnx.ONNX.ModelProto;
import onnx.ONNX.TensorProto;
import espam.utils.fileworker.ONNXFileWorker;

public class testOnnxConversion 
{
	public static void main(String args[]) 
	{
		
		String onnxpath = "oldfile.onnx";
		String onnxpathNew = "newfile.onnx";
		
		//Network network = CNNTestDataGenerator.createLenetSimple("simpleLeNet");
		//Tensor inputDataShapeExample = new Tensor(32,32,3);
		//network.setDataFormats(inputDataShapeExample);
		//ModelProto model = new GeneToOnnx(network).convertToONNXModel();
		//ONNXFileWorker.writeModel(model,onnxpath );
		
		//TESTING initializer modification below
		
		ModelProto model = ONNXFileWorker.readModel(onnxpath);
		ModelProto.Builder mb = model.toBuilder();
		GraphProto.Builder gb = mb.getGraphBuilder();
		List<TensorProto.Builder> tbList = gb.getInitializerBuilderList();
		
		Iterator<TensorProto.Builder> it = tbList.iterator();
		
		while(it.hasNext())
		{
			TensorProto.Builder tpb = it.next();
			System.out.println("Next initializer : "+tpb.getName() + ":" + tpb.getDimsCount());
			ByteString initializerData = tpb.getRawData();
			ByteBuffer bb = ByteBuffer.allocate(initializerData.size());
			
			initializerData.copyTo(bb);
			bb.order(ByteOrder.LITTLE_ENDIAN);
			
			bb.putFloat(0, bb.getFloat(0) + new Float(0.05));
			bb.putFloat(4, bb.getFloat(4) + new Float(0.05));
			bb.putFloat(8, bb.getFloat(8) + new Float(0.05));
			bb.rewind();
			
			tpb.setRawData(ByteString.copyFrom(bb, bb.capacity()));
	
			System.out.println("First 3 Values: " + bb.getFloat(0) + ":" + bb.getFloat(4) + ":" + bb.getFloat(8)) ;
		}
		
		ModelProto modelNew = mb.build();
		ONNXFileWorker.writeModel(modelNew,onnxpathNew );
		
		//List<TensorProto> inits = mb.getGraph().getInitializerList();//modelNew.getGraph().getInitializerList();
		
		
		
	}
}
