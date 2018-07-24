/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.languagedetector.helpers

import com.kotlinnlp.languagedetector.LanguageDetector
import com.kotlinnlp.languagedetector.dataset.Example
import com.kotlinnlp.simplednn.core.embeddings.EmbeddingsOptimizer
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adagrad.AdaGradMethod
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adam.ADAMMethod
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.dataset.Shuffler
import com.kotlinnlp.simplednn.deeplearning.attention.han.HANParameters
import com.kotlinnlp.simplednn.helpers.training.utils.ExamplesIndices
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.utils.progressindicator.ProgressIndicatorBar
import java.io.File
import java.io.FileOutputStream

/**
 * A helper for the training of a [LanguageDetector].
 *
 * @property languageDetector the [LanguageDetector] to train
 * @property epochs the number of training epochs
 * @property batchSize the size of each batch of examples (default = 1)
 * @property dropout the probability of dropout of the Embeddings (default = 0.0)
 * @property shuffler the [Shuffler] to shuffle the training sentences before each epoch (can be null)
 * @param paramsUpdateMethod the [UpdateMethod] for the neural parameters
 * @param embeddingsUpdateMethod the [UpdateMethod] for the embeddings
 */
class TrainingHelper(
  private val languageDetector: LanguageDetector,
  private val epochs: Int,
  private val batchSize: Int = 1,
  private val dropout: Double = 0.0,
  private val shuffler: Shuffler? = Shuffler(enablePseudoRandom = true, seed = 743),
  paramsUpdateMethod: UpdateMethod<*> = ADAMMethod(stepSize = 0.001),
  embeddingsUpdateMethod: UpdateMethod<*> = AdaGradMethod(learningRate = 0.1)
) {

  /**
   * The min absolute value that an output error must have in order to be propagated.
   */
  private val minRelevantError: Double = 1.0e-03

  /**
   * When timing started.
   */
  private var startTime: Long = 0

  /**
   * The best accuracy reached during the training.
   */
  private var bestAccuracy: Double = 0.0

  /**
   * The helper for the valdiation of the [languageDetector].
   */
  private val validationHelper = ValidationHelper(this.languageDetector)

  /**
   * The optimizer of the parameters of the [languageDetector].
   */
  private val optimizer: ParamsOptimizer<HANParameters> = ParamsOptimizer(
    params = this.languageDetector.model.params,
    updateMethod = paramsUpdateMethod)

  /**
   * The optimizer of the embeddings of the [languageDetector].
   */
  private val embeddingsOptimizer = EmbeddingsOptimizer(
    embeddingsMap = this.languageDetector.model.embeddings,
    updateMethod = embeddingsUpdateMethod)

  /**
   * Train the [languageDetector] using the given [trainingSet], validating each epoch if a [validationSet] is given.
   *
   * @param trainingSet the dataset to train the [languageDetector]
   * @param validationSet the dataset to validate the languageDetector after each epoch (default = null)
   * @param modelFilename the name of the file in which to save the best trained model (default = null)
   */
  fun train(trainingSet: ArrayList<Example>,
            validationSet: ArrayList<Example>? = null,
            modelFilename: String? = null) {

    this.initEmbeddings(trainingSet)

    (0 until this.epochs).forEach { i ->

      println("\nEpoch ${i + 1} of $epochs")

      this.startTiming()

      this.optimizer.newEpoch()
      this.embeddingsOptimizer.newEpoch()

      this.trainEpoch(trainingSet = trainingSet)

      println("Elapsed time: %s".format(this.formatElapsedTime()))

      if (validationSet != null) {
        this.validateAndSaveModel(validationSet = validationSet, modelFilename = modelFilename)
      }
    }
  }

  /**
   * Initialize the Embeddings of the [languageDetector] model, associating them to the chars contained in the given
   * [trainingSet].
   *
   * @param trainingSet the dataset to train the [languageDetector]
   */
  private fun initEmbeddings(trainingSet: ArrayList<Example>) {

    trainingSet.forEach { example ->
      example.text.forEach { char ->
        if (char !in this.languageDetector.model.embeddings) {
          this.languageDetector.model.embeddings.set(key = char)
        }
      }
    }
  }

  /**
   * Train the [languageDetector] on one epoch.
   *
   * @param trainingSet the training set
   */
  private fun trainEpoch(trainingSet: ArrayList<Example>) {

    val progress = ProgressIndicatorBar(trainingSet.size)

    this.optimizer.newBatch()
    this.embeddingsOptimizer.newBatch()

    for (exampleIndex in ExamplesIndices(size = trainingSet.size, shuffler = this.shuffler)) {

      progress.tick()

      this.optimizer.newExample()
      this.embeddingsOptimizer.newExample()

      this.learnFromExample(example = trainingSet[exampleIndex])

      if ((exampleIndex + 1) % this.batchSize == 0 || exampleIndex == trainingSet.lastIndex) {

        this.update()

        this.optimizer.newBatch()
        this.embeddingsOptimizer.newBatch()
      }
    }
  }

  /**
   * Learn from the given [example], comparing its gold output class with the one of the [languageDetector] and
   * accumulate the propagated errors.
   *
   * @param example the example from which to learn
   */
  private fun learnFromExample(example: Example) {

    this.languageDetector.forEachToken(example.text) { token ->

      val output: DenseNDArray = this.languageDetector.classifyToken(token, dropout = this.dropout)
      val goldClassIndex: Int = example.language.id

      val outputErrors = this.buildErrors(outputArray = output, goldClassIndex = goldClassIndex)

      if (this.errorsAreRelevant(outputErrors)) {
        this.languageDetector.backward(outputErrors = outputErrors)
        this.accumulateErrors(token)
      }
    }
  }

  /**
   * @param outputArray the output array of a prediction
   * @param goldClassIndex the expected class index
   *
   * @return the errors of the predicted [outputArray]
   */
  private fun buildErrors(outputArray: DenseNDArray, goldClassIndex: Int): DenseNDArray {

    val errors = outputArray.copy()

    errors[goldClassIndex] = errors[goldClassIndex] - 1

    return errors
  }

  /**
   * @param outputErrors the errors of an output array
   *
   * @return a Boolean indicating if the given [outputErrors] are relevant
   */
  private fun errorsAreRelevant(outputErrors: DenseNDArray): Boolean {

    return (0 until outputErrors.length).any { i -> Math.abs(outputErrors[i]) > this.minRelevantError }
  }

  /**
   * Accumulate classifier and input errors into the optimizers.
   *
   * @param token the input token
   */
  private fun accumulateErrors(token: String) {

    this.optimizer.accumulate(this.languageDetector.getParamsErrors(copy = false))

    val embeddingsErrors: ArrayList<DenseNDArray> = this.languageDetector.getInputSequenceErrors(copy = false)

    token.forEachIndexed { charIndex, char ->
      this.embeddingsOptimizer.accumulate(embeddingKey = char, errors = embeddingsErrors[charIndex])
    }
  }

  /**
   * Update all the parameters of the model.
   */
  private fun update() {

    this.optimizer.update()
    this.embeddingsOptimizer.update()
  }

  /**
   * Validate the [languageDetector] on the [validationSet] and save its best model to [modelFilename].
   *
   * @param validationSet the validation dataset to validate the [languageDetector]
   * @param modelFilename the name of the file in which to save the best model of the [languageDetector] (default null)
   */
  private fun validateAndSaveModel(validationSet: ArrayList<Example>, modelFilename: String?) {

    val accuracy = this.validateEpoch(validationSet = validationSet)

    println("Accuracy: %.2f%%".format(100.0 * accuracy))

    if (modelFilename != null && accuracy > this.bestAccuracy) {

      this.bestAccuracy = accuracy

      this.languageDetector.model.dump(FileOutputStream(File(modelFilename)))

      println("NEW BEST ACCURACY! Model saved to \"$modelFilename\"")
    }
  }

  /**
   * Validate the [languageDetector] after trained it on an epoch.
   *
   * @param validationSet the validation dataset to validate the [languageDetector]
   *
   * @return the current accuracy of the [languageDetector]
   */
  private fun validateEpoch(validationSet: ArrayList<Example>): Double {

    println("Epoch validation on %d sentences".format(validationSet.size))

    return this.validationHelper.validate(testSet = validationSet)
  }

  /**
   * Start registering time.
   */
  private fun startTiming() {
    this.startTime = System.currentTimeMillis()
  }

  /**
   * @return the formatted string with elapsed time in seconds and minutes.
   */
  private fun formatElapsedTime(): String {

    val elapsedTime = System.currentTimeMillis() - this.startTime
    val elapsedSecs = elapsedTime / 1000.0

    return "%.3f s (%.1f min)".format(elapsedSecs, elapsedSecs / 60.0)
  }
}
