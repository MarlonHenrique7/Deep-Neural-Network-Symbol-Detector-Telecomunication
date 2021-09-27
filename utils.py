import numpy as np
import math
import cmath
import csv
import time
import matplotlib.pyplot as plt
import os
from scipy.signal import hilbert
from scipy.spatial import distance
from scipy.special import erfc


def testar_gpu():
    train_on_gpu = torch.cuda.is_available() #Observa se a GPU está disponivel
    if train_on_gpu: #Se sim
        device = torch.device('cuda') #Seleciona o device como GPU
        print("Treinando na GPU") #E manda a mensagem
    else: #Se não
        device = torch.device('cpu') #Seleciona o device como cpu
        print("GPU indisponível, treinando na CPU") #E avisa que a GPU não esta disponível
    return device


def create_csv_rect(modelo, dict_SNRs):
    SNRs, num_exemplos = list(dict_SNRs.keys()), list(dict_SNRs.values())
    for i in range(len(SNRs)):
        arquivo = str(modelo.M) + '_QAM_rect' + str(SNRs[i]).replace('.',',') + '.csv' 
        with open(arquivo, 'w', newline='') as myfile:
            wr = csv.writer(myfile, delimiter=',', quoting=csv.QUOTE_ALL)
            for exemplos in range(num_exemplos[i]):
                snr = str(SNRs[i])
                modelo = M_QAM(modelo.M, bool_ruido=True, SNR_db=SNRs[i], Rfa_banda=modelo.Rfa_banda)
                modelo.cria_sinal()
                linha = np.concatenate(([str(modelo.simb_enviado)], [str(snr)], modelo.rect)).reshape(-1)
                wr.writerow(linha)

def create_csv_rect_embaralhado(modelo, dict_prob_SNRs, num_exemplos):
    SNRs, probs = list(dict_prob_SNRs.keys()), list(dict_prob_SNRs.values())
    arquivo = str(modelo.M) + '_QAM_rect' + 'embaralhado' + '.csv' 
    with open(arquivo, 'w', newline='') as myfile:
        wr = csv.writer(myfile, delimiter=',', quoting=csv.QUOTE_ALL)
        for exemplos in range(num_exemplos):
            snr = np.random.choice(SNRs, p=probs)
            modelo = M_QAM(modelo.M, bool_ruido=True, SNR_db=snr, Rfa_banda=modelo.Rfa_banda)
            modelo.cria_sinal()
            linha = np.concatenate(([str(modelo.simb_enviado)], [str(snr)], modelo.rect)).reshape(-1)
            wr.writerow(linha)

class Detector():
    def __init__(self, modelo):
        self.stars = np.zeros((2,modelo.M))
        i = 0
        for par in modelo.lista_par_ordenado:
            self.stars[:, i] = list(par)
            i += 1
        Estars = np.square(self.stars)[:][0]+np.square(self.stars)[:][1]
        self.a = (1/2)*(-Estars)

    def deteccao(self, entrada):
        entrada = np.array(entrada)
        x,y = np.mean(entrada[:,0]), np.mean(entrada[:,1])
        b = np.dot(np.transpose(self.stars), [x, y]) + self.a
        predict = np.argmax(b)
        return predict


class M_QAM():
    def __init__(self, M, bool_ruido=True, SNR_db=0, Rfa_banda = 30):
        self.M, self.SNR_db, self.Rfa_banda = M, SNR_db, Rfa_banda
        self.bool_ruido = bool_ruido
        self.n_bits = math.log2(M) #Número de bits por simbolo
        
        self.SNR = 10**(self.SNR_db/10)

        n_point_amos = int(M/4) #Número de pontos no primeiro quadrante
        dim = int(math.sqrt(n_point_amos)) #Número de pontos acessíveis em uma dimensão

        self.lista_dim = [] #Lista de apoio para guardar os valores possiveis na constelação
        for i in range(-dim,dim): #Loop entrando na quantidade de valores possiveis em uma dimensãp
            self.lista_dim.append((2*i + 1)/2) ##Lista que adiciona o valor correto que cada dimensão pode acessar

        self.lista_par_ordenado = [] #Lista para guardar os pares ordenados da constelação
        for j in self.lista_dim: #Loop que entra em cada valor possivel nas dimensões
            for h in self.lista_dim: #Novamente
                self.lista_par_ordenado.append((j,h)) #Lista que adiciona o par ordenado para cada um desses valores

        soma = 0 #Variável para guardar a soma das distancias de cada ponto
        count = 0
        for par in self.lista_par_ordenado: #Loop que percorre todos os pares ordenados
            if (par[0]>0) and (par[1]>0):
                soma += (distance.euclidean((0,0),par))**2 #Calcula a energia de cada um desses símbolos e soma 
                count += 1
        
        E_barra = self.Rfa_banda*soma/count #Tira a média pos símbolo
        N0 = E_barra/(self.n_bits*self.SNR)
        self.media_ruido, self.desvio_padrao_ruido = 0, math.sqrt(N0/2)

        amps_rect, fases_rect, dim_rect = [], [], []
        for par in self.lista_par_ordenado:
            dim1, dim2 = par[0], par[1]
            amps_rect.append(abs(dim1 + 1j*dim2))
            fases_rect.append(cmath.phase(dim1 + 1j*dim2)) 
            dim_rect.append(dim1)

        self.amp_max_rect, self.amp_min_rect = max(amps_rect)*1.1, min(amps_rect)*1.1 #1.1 é para alargar os limites para quando chegar o ruído
        self.fase_max_rect, self.fase_min_rect = max(fases_rect)*1.1, min(fases_rect)*1.1
        self.dim_max_rect, self.dim_min_rect = max(dim_rect)*1.1, min(dim_rect)*1.1

    def cria_sinal(self, random=True, sinal=0):
        self.simb_enviado = 0 #Lista que guardará os pares que definirão qual foi o símbolo enviado
        self.lista_ruido = []

        if random:
            par_escolhido = np.random.choice(self.M,1)
            par_escolhido = self.lista_par_ordenado[par_escolhido[0]]
        elif random==False:
            par_escolhido = self.lista_par_ordenado[sinal]


        self.dim1, self.dim2 = par_escolhido[0], par_escolhido[1]
        self.simb_enviado = (par_escolhido) #Adiciona-se os símbolos aleatórios na lista
        lista_dim1, lista_dim2 = self.dim1, self.dim2
        if self.bool_ruido:
            lista_dim1 = np.add(self.dim1,np.random.normal(self.media_ruido, self.desvio_padrao_ruido, self.Rfa_banda))
            lista_dim2 = np.add(self.dim2,np.random.normal(self.media_ruido, self.desvio_padrao_ruido, self.Rfa_banda))
        else:
            lista_dim1 = np.add(self.dim1,np.zeros(self.Rfa_banda))
            lista_dim2 = np.add(self.dim2,np.zeros(self.Rfa_banda))
        self.x, self.y = np.mean(lista_dim1), np.mean(lista_dim2)


        self.lista_dim1, self.lista_dim2 = lista_dim1, lista_dim2
        self.rect = lista_dim1 + 1j*lista_dim2 
        amp_rect = [abs(amostra) for amostra in self.rect]
        fase_rect = [cmath.phase(amostra) for amostra in self.rect]
        self.amp_rect, self.fase_rect = np.array(amp_rect), np.array(fase_rect)
        self.freq_rect = np.gradient(self.fase_rect)


    def constelacao(self):
        x2,y2 = self.simb_enviado 
        plt.scatter(self.x.real,self.y.real)
        plt.scatter(x2,y2,marker='*',c='red', edgecolors='face', s=100)
        plt.xlim(min(self.lista_dim)-1,max(self.lista_dim)+1)
        plt.ylim(min(self.lista_dim)-1,max(self.lista_dim)+1)
        plt.grid(True)
        plt.show()

    def constelacao_completa(self):
        for t in range(self.M):
            self.cria_sinal(random=False, sinal=t)
            x2,y2 = self.simb_enviado
            plt.scatter(x2,y2,marker='*',c='red', edgecolors='face', s=100)
            for u in range(self.Rfa_banda):
                self.cria_sinal(random=False, sinal=t)
                plt.scatter(self.x.real,self.y.real, c='black', edgecolors='face', s=2)
                
        plt.axhline(linewidth=0.4, color='black')
        plt.axvline(linewidth=0.4, color='black')
        plt.grid(True)
        plt.show()

    def P_erro_QAM(self, SNR):
        arg = math.sqrt((3*self.n_bits*SNR)/(self.M-1))
        Q = 0.5*erfc(arg/math.sqrt(2))
        P_erro = 4*(1-(1/math.sqrt(self.M)))*Q
        return P_erro

    def BER_teorico(self, SNRs_teorico):
        lista_erros_teoricos = []
        for snr in SNRs_teorico:
            snr = 10**(snr/10)
            P_erro_teorico = self.P_erro_QAM(snr)
            lista_erros_teoricos.append(P_erro_teorico)
        return SNRs_teorico, lista_erros_teoricos
