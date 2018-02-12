# メモ
https://web.stanford.edu/~hastie/Papers/mazumder10a.pdf
Soft_Imputeアルゴリズムと言うものがあって、低ランク行列の再構成が行える。

ちょっとこれを使って、分析をしようと思うから、下準備として用意し始める。

分析対象はこれ
https://www.kaggle.com/CooperUnion/anime-recommendations-database

# 概要
rating.csvのデータを特異値分解したところ、あまり低ランクとは言えない形だったので、
ちょっと頑張って行列に手を加えてみる。
