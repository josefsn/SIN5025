# from google_drive_downloader import GoogleDriveDownloader as gdd
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.inductive.variants.im_d import dfg_based as im_d_algo
from pm4py.algo.discovery.inductive.variants.im import algorithm as im_algo
from pm4py.algo.discovery.inductive.variants.im_f import algorithm as im_f_algo
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.evaluation.replay_fitness import evaluator as calc_fitness
from pm4py.evaluation.generalization import evaluator as calc_generaliz
from pm4py.evaluation.precision import evaluator as calc_precision
from pm4py.evaluation.simplicity import evaluator as calc_simplic
from pm4py.evaluation.soundness.woflan import algorithm as woflan

import tqdm
import datetime
import time
import pandas as pd
import os
import random


# def download_google_drive(google_id, fname):
#  gdd.download_file_from_google_drive(file_id=google_id, 
#                                      dest_path = './%s.xes'%fname, 
#                                      showsize = True)

def import_xes(log):
    return xes_importer.apply("%s.xes" % log)

def miner_model(log, miner, parameters=None):
    #model_res, ini_mark, fin_mark =
    return miner.apply(log, parameters=parameters)

def calc_metrics(log, petrinet_res, initial_mark=None, final_mark=None):
    def calc_and_time_precision():
        start_time = time.time()
        precision = calc_precision.apply(log, petrinet_res, initial_mark, final_mark)
        calc_duration = time.time() - start_time
        return round(precision, 4), round(calc_duration, 4)

    def calc_and_time_simplic():
        start_time = time.time()
        simplic = calc_simplic.apply(petrinet_res)
        calc_duration = time.time() - start_time
        return round(simplic, 4), calc_duration

    def calc_and_time_generaliz():
        start_time = time.time()
        generaliz = calc_generaliz.apply(log, petrinet_res, initial_mark, final_mark)
        calc_duration = time.time() - start_time
        return round(generaliz, 4), calc_duration

    def calc_and_time_fitness():
        start_time = time.time()
        fitness = calc_fitness.apply(log, petrinet_res, initial_mark, final_mark)
        calc_duration = time.time() - start_time
        return fitness, calc_duration

    def calc_fscore():
        try:
            fitness_avg = round(fitness['averageFitness'], 4)
        except:
            fitness_avg = round(fitness['average_trace_fitness'], 4)
        f_score = 2 * (fitness_avg * precision) / (fitness_avg + precision)
        return fitness_avg, f_score

    def calc_and_time_soudness():
        start_time = time.time()

        is_sound = woflan.apply(petrinet_res, initial_mark, final_mark,
                                parameters={woflan.Parameters.RETURN_ASAP_WHEN_NOT_SOUND: True,
                                                         woflan.Parameters.PRINT_DIAGNOSTICS: False,
                                                         woflan.Parameters.RETURN_DIAGNOSTICS: False})
        calc_duration = time.time() - start_time
        return str(is_sound), calc_duration

    precision, precision_time = calc_and_time_precision()
    simplic, simplic_time = calc_and_time_simplic()
    generaliz, generaliz_time = calc_and_time_generaliz()
    fitness, fitness_time = calc_and_time_fitness()
    fitness_avg, fscore = calc_fscore()
    sound, sound_time = calc_and_time_soudness()

    return (precision, precision_time, simplic, simplic_time,
            generaliz, generaliz_time, fitness_avg, fitness_time, fscore,
            sound, sound_time)

# set work dir
path = '/home/claudio/PycharmProjects/process_mining/exercicio2/'
os.chdir(path)
# set random seed
random.seed(20201101)


# Define logs
logs = {'domestic': {'google_id': 'DomesticDeclarations', 'petrinets': {}},
        'prepaid': {'google_id': 'PrepaidTravelCost', 'petrinets': {}},
        #'internacional': {'google_id': 'InternationalDeclarations', 'petrinets': {}},
        # 'permit': {'google_id': 'PermitLog', 'petrinets': {}},
        'request': {'google_id': 'RequestForPayment', 'petrinets': {}}
        }

# Define os algoritmo de descoberta de modelo de processo
miners = {'alpha': alpha_miner,
          'inductive': inductive_miner,
          'im': im_algo,
          'imf': im_f_algo,
          'imd': im_d_algo,
          'heuristic': heuristics_miner}

#miners = {'im': im_algo, 'imf': im_f_algo, 'imd': im_d_algo}

metrics = pd.DataFrame({'log': [], 'miner': [], 'miner_time': [], 'precision': [], 'precision_time': [],
                        'simplic': [], 'simplic_time': [], 'generaliz': [], 'generaliz_time': [],
                        'fitness': [], 'fitness_time': [], 'fscore': [], 'sound': [], 'sound_time': []})

# Para cada log
for log in logs:
    # Faz dowload dos dados
    # log_id = logs[log]['google_id']
    # download_google_drive(log_id, log)
    # Faz a leitura dos dados .xes para a estrutura de dados para log da PM4PY
    log_file = logs[log]['google_id']
    log_format = import_xes(log_file)
    logs[log]['log_format'] = log_format
    # Descobre modelo de processo para cada algoritmo de descoberta
    for miner in miners:
        miner_alg = miners[miner]
        miner_time = time.time()
        petrinet, initial_m, final_m = miner_model(log_format, miner_alg)
        miner_time = time.time() - miner_time
        logs[log]['petrinets'][miner] = petrinet
        #for log in logs:
        #    log_format = logs[log]['log_format']
        #    for miner in logs[log]['petrinets']:
        #petrinet = logs[log]['petrinets'][miner]
        log_metrics = calc_metrics(log_format, petrinet, initial_m, final_m)
        metrics.loc[len(metrics)] = [log, miner, miner_time] + list(log_metrics)
#metrics

metrics_file = 'run_metrics_rnd_' + str(datetime.datetime.now())
metrics.to_csv(metrics_file)
#metrics.to_excel(metrics_file)

