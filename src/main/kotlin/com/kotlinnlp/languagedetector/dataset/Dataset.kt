/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.languagedetector.dataset

/**
 * A dataset to train and test a LanguageDetector.
 *
 * @property training the training set as list of examples
 * @property validation the validation set as list of examples
 * @property test the test set as list of examples
 */
data class Dataset(val training: ArrayList<Example>, val validation: ArrayList<Example>, val test: ArrayList<Example>)
