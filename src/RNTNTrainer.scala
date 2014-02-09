import optiml.compiler._
import optiml.library._
import optiml.shared._

object RNTNInterpreter extends OptiMLApplicationInterpreter with RNTNTrainer
object RNTNCompiler extends OptiMLApplicationCompiler with RNTNTrainer
trait RNTNTrainer extends OptiMLApplication with RNTNOps with Utilities {
	private val TREECOLS = 5
	private val PARENT = 0  		// Gives the index of the parent node of the node
	private val LKID   = 1  		// Gives the index of the left child of the node
	private val RKID   = 2  		// Gives the index of the right child of the node
	private val WORD   = 3       	// Gives the phrase index represented by the node
	private val LABEL  = 4       	// Gives the predetermined sentiment score for the node
	//private val LEAFS  = 5       	// Gives the number of leaves the node is an ancestor of

	private val POSNEGNEUT = 5

	//private val QRNN  		  = true
	private val NUMCLASSES 	  = 5
	private val WORDSIZE      = 25

	private	val RUNSTHROUGHDATA = 20
	private val TRAINBATCHSIZE 	= 27
	private val EVALBATCHSIZE   = 192
	private val INITI 			= false

	private def readPhrases(filename: Rep[String]): Rep[DenseVector[DenseVector[String]]] = {
		val a = ForgeFileReader.readLines(filename){ line =>
        	val tokens = line.trim.fsplit(" ")
        	(0::array_length(tokens)) { d => tokens(d)}
      	}
      	(0::array_length(a)) {d => a(d)}
	}

	private def readIndices(filename: Rep[String]): Rep[IndexVector] = {
		readMatrix(filename, (e => e.toDouble), " ").getRow(0).find(e => e > 0)
	}

	// Read condensed matrix of trees and pick out individual trees using the fact that a node has
	// a parent entry of -1 iff it is a root of a tree
	private def readTrees( filename: Rep[String] ) = {
		println("Loading trees...")
		val rawTrees = readMatrix(filename + "_trees.csv");

		// Assumes that data is saved such that parents are always below their children
		val roots = rawTrees.findRows(d => d(PARENT) == 0)	// matlab data, 0 is invalid index
		val numTrees = roots.length
		val treeFirstLeaves = (0::1).toDense << (roots + 1)

		println("Pre-processing " + numTrees + " trees...")
		// Find all distinct words (NOT phrases) in tree
		val words = IndexVector(rawTrees.filterRows(d => d(LKID) == 0).getCol(WORD).map(d => d.toInt - 1).distinct)	// matlab data, 0 is invalid index
		val wordMapInit = DenseVector[Int](words.max + 1, true)
		wordMapInit.update(words, 1::words.length+1)	// Create word mapping 
		val wordMap = wordMapInit - 1
		
		val allTrees = (0::numTrees) { treeNum =>
			val curTree  = rawTrees(treeFirstLeaves(treeNum)::treeFirstLeaves(treeNum + 1))
			val numNodes = curTree.numRows
			val numLeafs = (numNodes + 1)/2

			val family = curTree.getCols(PARENT::LABEL).map(d => d.toInt - 1)
			val scores = curTree.getCol(LABEL)
			val levels = DenseVector[Int](numNodes, true)

			// Calculate level for each node
			var curNode = numNodes - 2
			while (curNode >= 0) {
				val parent = family(curNode, PARENT)
				val parentLevel = if (parent <= curNode || parent >= numNodes || parent < 0) {
	 				family.pprint
	 				errorOut("Error: There was a problem with the tree data: invalid parent given at node #" + curNode + " (requested node# " + parent + ")")
				}		
				else { levels(parent) }
				levels(curNode) = parentLevel + 1
				curNode -= 1
			}			

			val numLevels = levels.max + 1
			val nodesAtLevels = (0::numLevels) {curLevel => 
				levels.find(l => l == curLevel) 
			}

			// Form tree levels, output tree
			(0::numLevels) { curLevel =>
				val parents  = if (curLevel == 0) { nodesAtLevels(0) } else { nodesAtLevels(curLevel - 1) }
				val children = if (curLevel == numLevels - 1) { nodesAtLevels(0) } else { nodesAtLevels(curLevel + 1) } 
					
				(nodesAtLevels(curLevel), *) {node =>
					val origLeft = family(node, LKID)
					(0::TREECOLS) {col =>
						if (col == PARENT) {
							val origParent = family(node, PARENT)
							if (origParent != -1) {parents.find(d => d == origParent).first } else { -1 }
						} 
						else if (col == LKID) {
							if (origLeft != -1) { children.find(d => d == origLeft).first } else { -1 }
						}
						else if (col == RKID) { 
							val origChild = family(node, col)
							if (origChild != -1) { children.find(d => d == origChild).first } else { -1 }	
						}
						else if (col == WORD) {
							if (origLeft != -1) { -1 } else { wordMap(family(node, WORD)) }
						}
						else { // (col == LABEL) 
							val score = scores(node)
							if (NUMCLASSES <= 3) {
								if 	  	( score <= 0.4 ) 	{ 0 } 
					 		  	else if ( score >= 0.6 ) 	{ NUMCLASSES - 1 }
							  	else if ( NUMCLASSES == 3 ) { 1 }
							  	else		   			    { POSNEGNEUT } // neutral
							}
							else {
								if (score > 1) { errorOut("Error: An invalid phrase score of " + score + " exists in the tree at node" + node +".") }
								else {floor(abs(score - 0.001)*NUMCLASSES) }
							}
						}
					}
				}
			}
		}

		(allTrees, words)
		/*
		//if (fullTreesRaw.numCols != TREECOLS) {
		/*}	TODO: FIX LOADING PREVIOUSLY FORMATTED TREES
		else {
			val roots    = fullTreesRaw.getCol(PARENT).t.find(d => d < 0)
			val numTrees = roots.length
			roots <<= fullTreesRaw.numRows
			(0::numTrees) { treeNum => 
				fullTreesRaw.getRows(roots(treeNum)::roots(treeNum + 1)).map(_.toInt)
			}
		}*/*/
	}

	def main() = {
		val seed = 19542
		Global.randRef.setSeed(seed)

		val FILEPATH = "/home/david/PPL/data/rotten"
		val OUTPUT	 = "/home/david/PPL/outputs/"
		val DATASET = "/home/david/PPL/data/rotten"//if (args.length < 2) FILEPATH else args(0)

		// -------------------------------------------------------------------------------------//
		//								Load Data (preProDataset.m)					  			//
		// -------------------------------------------------------------------------------------//
		println("Loading data...")
		tic("Transform trees")
		// Vector of matrices. Each matrix represent a full tree. Trees have numbered nodes
		val (allTrees, words) = readTrees(DATASET)			
		toc("Transform trees")
		
		// Vector of vector of strings representing parsed sentences
		val parsedPhrases = readPhrases(DATASET + "_words.csv")

		val trainIndices = readIndices(DATASET + "_trainIndices.csv")
		val testIndices  = readIndices(DATASET + "_testIndices.csv")
		val devIndices   = readIndices(DATASET + "_devIndices.csv")

		val numWords = words.length
		val numTrees = allTrees.length

		val trainTrees = allTrees(trainIndices)
		
		val (trainBatches, trainWords) = createEvalBatches(trainTrees, EVALBATCHSIZE)
		val (devBatches, devWords) = createEvalBatches(allTrees(devIndices), EVALBATCHSIZE)
		val (testBatches, testWords) = createEvalBatches(allTrees(testIndices), EVALBATCHSIZE)

		println("Data loaded. " + numTrees + " phrase trees with " + numWords + " total words")
		// -------------------------------------------------------------------------------------//
		//								Init Weights (intParams.m)						  		//
		// -------------------------------------------------------------------------------------//
		println("Initializing weights...")
		
		val fanIn   = WORDSIZE.toDouble
		val range   = 1/sqrt(fanIn)
		val rangeW  = 1/sqrt(2*fanIn)
		val rangeWt = 1/sqrt(4*fanIn)
		val sizeWt  = WORDSIZE*2

		// NUMCLASSES X (WORDSIZE + 1)
		val WcF = (0::NUMCLASSES, 0::WORDSIZE + 1) {(row, col) => 
			if 		(col < WORDSIZE - 1) 	{ random[Double] * (2*range) - range }
			else if (col == WORDSIZE - 1)	{ random[Double] }
			else 							{ 0.0 }
		}
		
		// WORDSIZE X (WORDSIZE * 2 + 1)
		val WF = (0::WORDSIZE, 0::(WORDSIZE*2 + 1)) {(row, col) => 
			if (col < WORDSIZE*2) {
				if (INITI) { if (row == col || (row + WORDSIZE) == col) 0.5 else 0.0 }
				else	   { random[Double] * (2*rangeW) - rangeW }
			}
			else { 0.0 }
		}

		// WORDSIZE X (WORDSIZE * 2) X (WORDSIZE * 2)
		val WtF = DenseMatrix.randn(WORDSIZE*sizeWt, sizeWt) * (2*rangeWt) - rangeWt

		// weights associated with individual phrases (each row belongs to a word)
		val WvF = DenseMatrix.randn(numWords, WORDSIZE) * 0.1

		val Wc = WcF.mutable
		val W  = WF.mutable
		val Wt = WtF.mutable
		val Wv = WvF.mutable

		val ssWc = DenseMatrix[Double](Wc.numRows, Wc.numCols)
		val ssW  = DenseMatrix[Double](W.numRows,  W.numCols)
		val ssWt = DenseMatrix[Double](Wt.numRows, Wt.numCols)
		val ssWv = DenseMatrix[Double](Wv.numRows, Wv.numCols)

		writeMatrix(Wc, OUTPUT + "Wc_init.txt")
		writeMatrix(W,  OUTPUT + "W_init.txt")
		writeMatrix(Wt, OUTPUT + "Wt_init.txt")
		writeMatrix(Wv, OUTPUT + "Wv_init.txt")

		println("Weight initialization completed")
		// -------------------------------------------------------------------------------------//
		//								Run Training and Evaluation						  		//
		// -------------------------------------------------------------------------------------//
		val numTrainBatches = ceil(trainTrees.length.toDouble/TRAINBATCHSIZE)
		println("Training " + trainTrees.length + " training set trees in " + numTrainBatches + " batches")

	   	var runIter = 1
	   	while (runIter <= RUNSTHROUGHDATA) {
			//println("Training run " + runIter + "/" + RUNSTHROUGHDATA)
			//trainOnTrees(trainTrees, Wc, W, Wt, Wv, ssWc, ssW, ssWt, ssWv, TRAINBATCHSIZE, numTrainBatches)
			//println("Completed run " + runIter + "/" + RUNSTHROUGHDATA)
			//println("")
			println(" --------------------- Train Set Accuracy (After " + runIter + "/" + RUNSTHROUGHDATA + ") --------------------- ")
			evalOnTrees(trainBatches, trainWords, Wc, W, Wt, Wv)
			println("-----------------------------------------------------------------------------")
			
	   		println(" --------------------- Dev Set Accuracy (After " + runIter + "/" + RUNSTHROUGHDATA + ") --------------------- ")
			evalOnTrees(devBatches, devWords, Wc, W, Wt, Wv)
			println("-----------------------------------------------------------------------------")
			println("")
			//println("Writing out results...")
			//writeMatrix(Wc, OUTPUT + "Wc_run" + runIter + ".txt")
			//writeMatrix(W,  OUTPUT + "W_run" + runIter + ".txt")
			//writeMatrix(Wt, OUTPUT + "Wt_run" + runIter + ".txt")
			//writeMatrix(Wv, OUTPUT + "Wv_run" + runIter + ".txt")
			runIter += 1
		}		

		println("-------------------------- Test Set Final Accuracy --------------------------")
		evalOnTrees(testBatches, testWords, Wc, W, Wt, Wv)
		println("-----------------------------------------------------------------------------")
	}	// end of main

}