import motmetrics.metrics
import pandas


def idf1(num_objects, num_predictions):
    pd = pandas.DataFame()
    glob_assig = motmetrics.metrics.id_global_assignment(pd)
    idfn = motmetrics.metrics.idfn(pd, glob_assig)
    idtp = motmetrics.metrics.idtp(pd, glob_assig, num_objects, idfn)

    idf1_value = motmetrics.metrics.idf1(pd, idtp, num_objects, num_predictions)
    return idf1_value
