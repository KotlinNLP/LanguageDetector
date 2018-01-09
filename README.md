# LanguageDetector [![GitHub version](https://badge.fury.io/gh/KotlinNLP%2FLanguageDetector.svg)](https://badge.fury.io/gh/KotlinNLP%2FLanguageDetector) [![Build Status](https://travis-ci.org/KotlinNLP/LanguageDetector.svg?branch=master)](https://travis-ci.org/KotlinNLP/LanguageDetector)

LanguageDetector is a very simple to use text language detector which uses the Hierarchical Attention Networks (HAN) from the [SimpleDNN](https://github.com/nlpstep/simplednn "SimpleDNN") library.

LanguageDetector is part of [KotlinNLP](http://kotlinnlp.com/ "KotlinNLP").


## Getting Started

### Import with Maven

```xml
<dependency>
    <groupId>com.kotlinnlp</groupId>
    <artifactId>languagedetector</artifactId>
    <version>0.1.0</version>
</dependency>
```

### Examples

Try some examples of usage of LanguageDetector running the files in the `examples` folder.

To run the examples you need datasets of test and training that you can find
[here](https://www.dropbox.com/ "LanguageDetector examples datasets")

### Model Serialization

The trained model is all contained into a single class which provides simple dump() and load() methods to serialize it and afterwards load it.


## License

This software is released under the terms of the 
[Mozilla Public License, v. 2.0](https://mozilla.org/MPL/2.0/ "Mozilla Public License, v. 2.0")


## Contributions

We greatly appreciate any bug reports and contributions, which can be made by filing an issue or making a pull 
request through the [github page](https://github.com/nlpstep/LanguageDetector "LanguageDetector on GitHub").
