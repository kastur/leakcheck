package edu.ucla.nesl.privacy;

import java.util.HashSet;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.attribute.Remove;

public class LeakCheck {

  // Evaluates a classifier using cross-validation on some instances. The input attributes are input_inds, and the output attribute is class_ind.
	public static Evaluation doQuickEvaluate(Classifier classifier, Instances data, HashSet<Integer> input_inds, int class_ind) throws Exception {
		data.setClassIndex(class_ind);
		int[] remove_inds = new int[data.numAttributes() - input_inds.size() - 1];
		int ii = 0;
		for (int ind = 0; ind < data.numAttributes(); ++ind) {
			if (ind != class_ind && !input_inds.contains(ind))
				remove_inds[ii++] = ind;
		}
		Remove remove = new Remove();
		remove.setAttributeIndicesArray(remove_inds);
		remove.setInputFormat(data);
		
		FilteredClassifier filteredClassifier = new FilteredClassifier();
		filteredClassifier.setFilter(remove);
		filteredClassifier.setClassifier(classifier);
		 
		filteredClassifier.buildClassifier(data);
		
		 Evaluation eval = new Evaluation(data);
		 eval.crossValidateModel(filteredClassifier, data, 3, new Random(1));
		 return eval;
	}
	
	public static void main(String[] args) throws Exception {
		String arffFileName = "../data/iris.data.prog.arff";
		DataSource source = new DataSource(arffFileName);
		
		Instances data = source.getDataSet();
		
		for (int ii = 0; ii < data.numAttributes(); ++ii) {
			System.out.println(
					"Name: " + data.attribute(ii).name() +
					", Index: " + data.attribute(ii).index());
		}
		
		HashSet<Integer> input_inds = new HashSet<Integer>();

    // progA and progB are the input attributes.
		input_inds.add(5);
		input_inds.add(6);
	
    // Evaluate how accurately we can infer the remaining attributes
    // given progA and progB as the inputs.

    // The 4 numeric attributes, we use a lienar regression model.
		{
			LinearRegression linreg = new LinearRegression();
			Evaluation eval = doQuickEvaluate(linreg, data, input_inds, 0);
			System.out.println(eval.pctCorrect() + "," + eval.rootMeanSquaredError());
		}
		
		{
			LinearRegression linreg = new LinearRegression();
			Evaluation eval = doQuickEvaluate(linreg, data, input_inds, 1);
			System.out.println(eval.pctCorrect() + "," + eval.rootMeanSquaredError());
		}
		
		{
			LinearRegression linreg = new LinearRegression();
			Evaluation eval = doQuickEvaluate(linreg, data, input_inds, 2);
			System.out.println(eval.pctCorrect() + "," + eval.rootMeanSquaredError());
		}
		
		{
			LinearRegression linreg = new LinearRegression();
			Evaluation eval = doQuickEvaluate(linreg, data, input_inds, 2);
			System.out.println(eval.pctCorrect() + "," + eval.rootMeanSquaredError());
		}

    // The last iris flower "class" attribute we use a J48 decision tree.
		{
			J48 j48 = new J48();
			Evaluation eval = doQuickEvaluate(j48, data, input_inds, 4);
			System.out.println(eval.pctCorrect() + "," + eval.rootMeanSquaredError());
		}
		

    // Sanity check :::
    // Also evaluate how accurately we can infer the iris flower "class"
    // given the four numeric attributes the input.
		System.out.println("{0,1,2,3} => {4}");
		HashSet<Integer> input_inds_2 = new HashSet<Integer>();
		input_inds_2.add(0);
		input_inds_2.add(1);
		input_inds_2.add(2);
		input_inds_2.add(3);
		{
			J48 j48 = new J48();
			Evaluation eval = doQuickEvaluate(j48, data, input_inds_2, 4);
			System.out.println(eval.pctCorrect() + "," + eval.rootMeanSquaredError());
		}

    System.out.println("DONE!");
	}
}
