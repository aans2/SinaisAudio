#Base de Dados
##Opção escolhida: GTZAN (a)

##Justificativa: O dataset GTZAN é amplamente utilizado em pesquisas de classificação de gêneros musicais, oferecendo uma variedade de gêneros e uma estrutura padronizada que facilita o treinamento e a validação do modelo.

#Pré-processamento
##Opções escolhidas:

a. Repartição em janelas de ≈20ms (janelamento)

b. Window functions (por exemplo, Hamming contido na lista)

c. Sobreposição de janelas

##Justificativa:

O janelamento com aproximadamente 20ms permite a análise de características locais do sinal, que é crucial para sinais de áudio.

O uso de window functions (como a Hamming) ajuda a reduzir efeitos indesejados nas bordas de cada janela, suavizando o sinal.

A sobreposição de janelas garante a continuidade entre as análises sucessivas, permitindo capturar transições e detalhes que poderiam ser perdidos com janelas não sobrepostas.

#Extração de Parâmetros
##Procedimentos escolhidos:

Zero Crossing Rate (ZCR)

Categoria: Temporais (opção a.ii)

Justificativa: O ZCR fornece informações sobre a frequência das mudanças de sinal, ajudando a diferenciar características de instrumentos ou vocais.

Spectral Centroid

Categoria: Espectrais (opção b.ii)

Justificativa: O Spectral Centroid é calculado a partir da Transformada de Fourier, fornecendo uma medida da "brilhância" ou centro de massa do espectro. Isso cumpre a exigência de incluir um método baseado na FFT.

Mel-frequency cepstral coefficients (MFCC) e suas derivadas (ΔMFCC e ΔΔMFCC)

Categoria: Cepstrais (opção c.i)

Justificativa: Os MFCCs são amplamente utilizados na análise de áudio, pois capturam informações relevantes à percepção humana. Eles são computados utilizando a FFT, atendendo também à exigência de métodos com base na Transformada de Fourier.

#Reconhecedores
##Opção escolhida: Support Vector Machines (SVM) (d)

##Justificativa:
O SVM é um classificador robusto que tem se mostrado eficaz em tarefas de classificação com dados de alta dimensão, como os provenientes da extração de características de áudio. Além disso, sua capacidade de generalização é adequada para cenários com conjuntos de dados relativamente padronizados, como o GTZAN.