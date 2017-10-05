/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

import com.kotlinnlp.languagedetector.dataset.CorpusReader
import com.kotlinnlp.languagedetector.dataset.Example
import com.kotlinnlp.languagedetector.utils.FrequencyDictionary
import com.kotlinnlp.languagedetector.utils.TextTokenizer
import com.kotlinnlp.neuraltokenizer.NeuralTokenizerModel
import com.kotlinnlp.simplednn.utils.progressindicator.ProgressIndicatorBar
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream

/**
 * Save the words frequency per language into a [FrequencyDictionary] reading them from a dataset and save the
 * serialized model of the dictionary to file.
 *
 * The first argument is the serialized model of the NeuralTokenizer for Chinese, Japanese and Korean texts.
 * The second argument is the filename of the input dataset.
 * The third argument is the output filename of the serialized model of the dictionary.
 */
fun main(args: Array<String>) {

  println("Loading CJK NeuralTokenizer model from '${args[0]}'...")
  val cjkModel = NeuralTokenizerModel.load(FileInputStream(File(args[0])))
  val tokenizer = TextTokenizer(cjkModel)

  println("Reading dataset from '${args[1]}'...")
  val dataset: ArrayList<Example> = CorpusReader().read(File(args[1]))

  val dictionary = FrequencyDictionary()
  val fileSize: Int = File(args[1]).getNumOfLines()
  val progress = ProgressIndicatorBar(fileSize)

  println("Counting words occurrences...")

  dataset.forEach { (text, language) ->

    progress.tick()

    tokenizer.tokenize(text, maxTokensLength = 100).forEach { token ->
      dictionary.addOccurrence(word = token, language = language)
    }
  }

  dictionary.normalize()

  println("Saving dictionary to '${args[2]}'...")
  dictionary.dump(FileOutputStream(File(args[2])))
}
