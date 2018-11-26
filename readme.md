## Introduction


In a recent era of big data, text contents are being produced much faster than a human being can consume. According to [Marr](https://www.forbes.com/sites/bernardmarr/2018/05/21/how-much-data-do-we-create-every-day-the-mind-blowing-stats-everyone-should-read/#330c8ef360ba), as of May 2018, there are 16 million text messages, 156 million emails and 456,000 tweets being sent every minute. In addition, modern technologies (e.g. the Internet) produce news in different formats, such as text messages, social media, online subscription... Obviously, we are not able to consume every single news article in its original form. For readers who would prefer to grasp the main ideas, It’s much more efficient to summarize these news articles into shorter texts. However, manual text summarization is tedious and laborious.


With the power of computer, we hope to perform the text summarization task automatically. Automatic text summarization has several advantages over the manual approach, for instances fewer biases and more personalized recommendations. Since news articles are more organized, it is easier to summarize them in a meaningful way. Our team explored several latest methodologies for automatic text summarization and apply it on a large-scale dataset, namely [DeepMind Q&A](https://cs.nyu.edu/~kcho/DMQA/) shared by Google. This dataset contains around 90,000 documents from CNN news. Each document is composed of the body text and human summarized “highlights”. Our goal is to construct a summary comparable with the “highlights” given a news body.


## Usage


First, you may need to download the DMQA dataset and put them under `cnn_stories` folder. Following 4 methods are implemented. Please refer to the codes about how to use them.


1. TF-IDF tag based method
2. Modified LexRank
3. Latent Semantic Analysis
4. NetSum



