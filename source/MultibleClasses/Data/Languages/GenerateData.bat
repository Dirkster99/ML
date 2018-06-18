@ECHO OFF
REM
REM
REM
type 01_Train\deu_mixed-typical_2011_10K-sentences.txt   > 01_Train.txt
type 01_Train\eng_news_2005_10K-sentences.txt           >> 01_Train.txt
type 01_Train\fra_mixed_2009_10K-sentences.txt          >> 01_Train.txt
type 01_Train\ita_mixed-typical_2017_10K-sentences.txt  >> 01_Train.txt
type 01_Train\ron_news_2015_10K-sentences.txt           >> 01_Train.txt
type 01_Train\spa_news_2006_10K-sentences.txt           >> 01_Train.txt
REM
REM
REM
type 02_Test\deu_mixed-typical_2011_10K-sentences.txt  > 02_Test.txt
type 02_Test\eng_news_2005_10K-sentences.txt          >> 02_Test.txt
type 02_Test\fra_mixed_2009_10K-sentences.txt         >> 02_Test.txt
type 02_Test\ita_mixed-typical_2017_10K-sentences.txt >> 02_Test.txt
type 02_Test\ron_news_2015_10K-sentences.txt          >> 02_Test.txt
type 02_Test\spa_news_2006_10K-sentences.txt          >> 02_Test.txt
