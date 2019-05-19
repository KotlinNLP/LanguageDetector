/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.languagedetector.utils

import com.kotlinnlp.neuraltokenizer.NeuralTokenizer
import com.kotlinnlp.neuraltokenizer.NeuralTokenizerModel
import java.io.File

/**
 * A simple tokenizer which splits a text by spacing and pointing chars.
 *
 * @param cjkModel the model of the [NeuralTokenizer] used to tokenize Chinese, Japanese and Korean texts
 */
class TextTokenizer(cjkModel: NeuralTokenizerModel) {

  companion object {

    /**
     * The min percentage of chars that a token must have to be considered Chinese, Japanese or Korean
     */
    private const val MIN_CJK_CHARS_PERCENTAGE = 0.40
  }

  /**
   * The currently buffered token.
   */
  private val tokenBuffer = StringBuffer()

  /**
   * The list of tokens that is filled during the tokenization.
   */
  private val tokens = mutableListOf<String>()

  /**
   * A set of the most frequent Chinese, Japanese and Korean characters.
   */
  private val cjkChars: Set<Char> = setOf(
    *this.javaClass.getResourceAsStream(File.separator + "CJKChars.txt").reader().readText()
      .split("\n")
      .filter { it.isNotEmpty() }
      .map { it[0] }
      .toTypedArray()
  )

  /**
   * The [NeuralTokenizer] for Chinese, Japanese and Korean texts.
   */
  private val cjkNeuralTokenizer = NeuralTokenizer(cjkModel)

  /**
   * Tokenize a text by spacing an punctuation chars.
   *
   * @param text the text to tokenize
   * @param maxTokensLength the max length of a token (longer tokens will be split)
   *
   * @return the list of tokens as list of [String]s
   */
  fun tokenize(text: String, maxTokensLength: Int): List<String> {
    require(maxTokensLength > 0)

    this.resetBuffers()

    text.forEach { this.processChar(char = it, maxTokensLength = maxTokensLength) }

    if (this.tokenBuffer.isNotEmpty()) {
      this.addToken()
    }

    return this.tokenizeCJKTokens()
  }

  /**
   * Process the given [char] of the input text.
   * If it is a spacing or punctuation char the current buffered token is added to the [tokens] list. Otherwise the
   * char is added to the buffered token.
   *
   * @param char a Char of the processing string
   * @param maxTokensLength the max length of a token
   */
  private fun processChar(char: Char, maxTokensLength: Int) {

    if (!char.isLetter()) {

      this.addToken()

    } else {

      this.tokenBuffer.append(char)

      if (this.tokenBuffer.length >= maxTokensLength) {
        this.addToken()
      }
    }
  }

  /**
   * Add the currently buffered token to the list of [tokens] and reset it.
   */
  private fun addToken() {

    val token: String = this.tokenBuffer.toString()

    if (token.isNotEmpty()) {
      this.tokens.add(token)
    }

    this.tokenBuffer.setLength(0)
  }

  /**
   * Reset all buffers (current token and tokens list).
   */
  private fun resetBuffers() {

    this.tokens.clear()
    this.tokenBuffer.setLength(0)
  }

  /**
   * Tokenize all [tokens] that are identified as Chinese, Japanese or Korean, returning a more complete list.
   * This is necessary because often spacing chars are not used in those languages.
   *
   * @return a List of tokens ([String]s)
   */
  private fun tokenizeCJKTokens(): List<String> {

    val finalTokens = mutableListOf<String>()

    this.tokens.forEach { token ->

      if (token.isCJK()) {
        cjkNeuralTokenizer.tokenize(token).forEach { sentence -> sentence.tokens.forEach { finalTokens.add(it.form) } }

      } else {
        finalTokens.add(token)
      }
    }

    return finalTokens.toList()
  }

  /**
   *
   */
  private fun String.isCJK(): Boolean {

    val cjkCharsCount = this.sumBy { if (it in this@TextTokenizer.cjkChars) 1 else 0 }

    return (cjkCharsCount.toDouble() / this.length) >= MIN_CJK_CHARS_PERCENTAGE
  }
}
