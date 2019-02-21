/* Copyright 2016-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package dictionary

import com.xenomachina.argparser.ArgParser

/**
 * The interpreter of command line arguments.
 *
 * @param args the array of command line arguments
 */
internal class CommandLineArguments(args: Array<String>) {

  /**
   * The parser of the string arguments.
   */
  private val parser = ArgParser(args)

  /**
   * The file path of the Chinese-Japanese-Korean (CJK) tokenizer serialized model.
   */
  val cjkTokenizerModelPath: String by parser.storing(
    "-c",
    "--cjk-tokenizer",
    help="the file path of the Chinese-Japanese-Korean (CJK) tokenizer serialized model"
  )

  /**
   * The path of the input JSONL file.
   */
  val inputFilePath: String by parser.storing(
    "-i",
    "--input",
    help="the path of the input JSONL file"
  )

  /**
   * The path of the output serialized dictionary.
   */
  val outputFilePath: String by parser.storing(
    "-o",
    "--output",
    help="the path of the output serialized dictionary"
  )

  /**
   * Force parsing all arguments (only read ones are parsed by default).
   */
  init {
    parser.force()
  }
}
