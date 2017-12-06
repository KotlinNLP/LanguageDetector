/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.languagedetector

import com.kotlinnlp.languagedetector.utils.Language
import com.kotlinnlp.simplednn.core.functionalities.activations.Softmax
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.han.HAN
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.han.HANParameters
import com.kotlinnlp.simplednn.deeplearning.embeddings.EmbeddingsMap
import com.kotlinnlp.simplednn.utils.Serializer
import java.io.*

/**
 * The model of a [LanguageDetector].
 *
 * @property embeddingsSize the size of the embeddings (default = 50)
 * @property attentionSize the size of the attention arrays of the HAN Encoder (default = 50)
 * @property hiddenSize the size of the hidden layer of the HAN Encoder (default = 150)
 * @property maxTokensLength the max length of a token, longer tokens will be split (default = 100)
 * @param recurrentConnectionType the connection type of the recurrent neural networks
 */
class LanguageDetectorModel(
  val embeddingsSize: Int = 50,
  val attentionSize: Int = 50,
  val hiddenSize: Int = 150,
  val maxTokensLength: Int = 100,
  recurrentConnectionType: LayerType.Connection = LayerType.Connection.RAN
) : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed from Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L

    /**
     * Read a [LanguageDetectorModel] (serialized) from an input stream and decode it.
     *
     * @param inputStream the [InputStream] from which to read the serialized [LanguageDetectorModel]
     *
     * @return the [LanguageDetectorModel] read from [inputStream] and decoded
     */
    fun load(inputStream: InputStream): LanguageDetectorModel = Serializer.deserialize(inputStream)
  }

  /**
   * The list of all supported languages (the index of a [Language] is equal to its id).
   * The Unknown language is excluded.
   */
  val supportedLanguages: List<Language> = Language.values().slice(0 until Language.values().size - 1)

  /**
   * The map of chars to embeddings.
   */
  val embeddings = EmbeddingsMap<Char>(size = embeddingsSize)

  /**
   * The [HAN] model.
   */
  val han = HAN(
    hierarchySize = 1,
    inputSize = this.embeddingsSize,
    inputType = LayerType.Input.Dense,
    attentionSize = this.attentionSize,
    outputSize = this.supportedLanguages.size,
    outputActivation = Softmax(),
    biRNNsActivation = Tanh(),
    biRNNsConnectionType = recurrentConnectionType,
    gainFactors = arrayOf(this.hiddenSize.toDouble() / this.embeddingsSize))

  /**
   * The parameters of the model.
   */
  val params: HANParameters = this.han.params

  /**
   * Serialize this [LanguageDetectorModel] and write it to an output stream.
   *
   * @param outputStream the [OutputStream] in which to write this serialized [LanguageDetectorModel]
   */
  fun dump(outputStream: OutputStream) = Serializer.serialize(this, outputStream)

  /**
   * @return the string representation of this model
   */
  override fun toString(): String {

    return  """
      - embeddings size: %d
      - attention size: %d
      - hidden layer size: %d
      - max tokens length: %d
      - recurrent connection type: %s
    """
      .format(
        this.embeddingsSize,
        this.attentionSize,
        this.hiddenSize,
        this.maxTokensLength,
        this.han.biRNNsConnectionType.name
      )
      .trimIndent()
  }
}
