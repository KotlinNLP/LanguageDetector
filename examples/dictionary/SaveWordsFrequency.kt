/* Copyright 2016-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package dictionary

import com.kotlinnlp.languagedetector.dataset.CorpusReader
import com.kotlinnlp.languagedetector.dataset.Example
import com.kotlinnlp.languagedetector.utils.FrequencyDictionary
import com.kotlinnlp.languagedetector.utils.TextTokenizer
import com.kotlinnlp.neuraltokenizer.NeuralTokenizerModel
import com.kotlinnlp.utils.getLinesCount
import com.kotlinnlp.utils.progressindicator.ProgressIndicatorBar
import com.xenomachina.argparser.mainBody
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream

/**
 * Save the words frequency per language into a [FrequencyDictionary] reading them from a dataset and save the
 * serialized model of the dictionary to file.
 *
 * Launch with the '-h' option for help about the command line arguments.
 */
fun main(args: Array<String>) = mainBody {

  val parsedArgs = CommandLineArguments(args)

  val tokenizer = TextTokenizer(
    cjkModel = parsedArgs.cjkTokenizerModelPath.let {
      println("Loading CJK tokenizer model from '$it'...")
      NeuralTokenizerModel.load(FileInputStream(File(it)))
    })
  val dataset: ArrayList<Example> = parsedArgs.inputFilePath.let {
    println("Reading dataset from '$it'...")
    CorpusReader().read(File(it))
  }
  val dictionary = FrequencyDictionary()
  val fileSize: Int = getLinesCount(parsedArgs.inputFilePath)
  val progress = ProgressIndicatorBar(fileSize)

  println("Counting words occurrences...")

  dataset.forEach { (text, language) ->

    progress.tick()

    tokenizer.tokenize(text, maxTokensLength = 100).forEach { token ->
      dictionary.addOccurrence(word = token, language = language)
    }
  }

  dictionary.normalize()

  println("Saving dictionary to '${parsedArgs.outputFilePath}'...")
  dictionary.dump(FileOutputStream(File(parsedArgs.outputFilePath)))
}
