import os
import pandas as pd
import pymongo
import getopt
import sys
from pymongo import MongoClient
import bson

class mongo_api:
    def __init__(self, host='localhost', port=27017, database_name = 'trademl'):
        self.client = MongoClient(host,int(port))
        self.db = self.client[database_name]
                       
        
    def write_df(self, my_df, collection_name,
                          chunk_size = 100):
        collection = self.db[collection_name]
        # To write
        ##collection.delete_many({})  # Destroy the collection
        #aux_df=aux_df.drop_duplicates(subset=None, keep='last') # To avoid repetitions
        my_list = my_df.to_dict('records')
        l =  len(my_list)
        if l > 0:
            collection.insert_many(my_list) # fill de collection
        #print('Done')
        return l

    def deleteMany(self, collection_name, filter):
        collection = self.db[collection_name]
        return collection.delete_many(filter)

    @staticmethod
    def isEmpty(obj):
        if obj is Null:
            return True;
        for prop in obj:
            if obj.hasOwnProperty(prop):
                return False
        return True

    @staticmethod
    def generateAggrColumns(projectionAttrs, projectionMeasures, filter, sortSpec):
        idColumns = {}
        groupColumns = {}
        projects = {}
        for c in projectionAttrs:
            value = None
            key = c
            if type(c) is dict:
                values = c.values()
                keys = c.keys()
                for k in keys:
                    value = c[k]
                    idColumns[k] = value
                    accValue = {'$first': value}
                    groupColumns[k] = accValue
                    projects[k] = 1
            else:
                value = "$" + c
                idColumns[key] = value
                accValue = {'$first': value}
                groupColumns[key] = accValue
                projects[key] = 1
        groupColumns["_id"] = idColumns
        projects["_id"] = 0
        addFields = {}
        if len(projectionMeasures) > 0:
            for c in projectionMeasures:
                value = "$" + c
                accValue = {'$avg': value}
                fieldName = "avg_" + c
                groupColumns[fieldName] = accValue
                projects[fieldName] = 1
                roundField = "$" + fieldName
                addFields[fieldName]={'$round': [roundField, 2]}
        pipeline = []
        pipeline.append({'$match': filter})
        pipeline.append({'$group': groupColumns})
        pipeline.append({'$project': projects})
        pipeline.append({'$addFields': addFields})
        pipeline.append({'$sort': sortSpec})
        print(pipeline)
        return pipeline

    def count(self, collection_name, filter):
        try:
            collection = self.db[collection_name]
            if collection is not None:
                return collection.find(filter).count()
            else:
                return 0
        except bson.errors.InvalidDocument as Error:
            print("Error on filter")
            exit(-1)

    def countDistinct(self, collection_name, projection, filter):
        collection = self.db[collection_name]
        if collection is not None:
            return len(collection.find(filter).distinct(projection))
        else:
            return 0

    def getProjection(self, collection_name):
        collection = self.db[collection_name]
        cursor = collection.find({}).limit(1)
        result = None
        if cursor is not None:
            for n in cursor:
                return list(n.keys())


    def read_df(self,  collection_name, distinct, projectionAttrs, projectionMeasures, filter, sortSpec):
        collection = self.db[collection_name]
        cursor = None
        result = []
        if (distinct is False) and len(projectionMeasures) == 0:
            projection = projectionAttrs
            if filter is not {}:
                cursor = collection.find(filter, projection)
            else:
                cursor = collection.find({}, projection)

        else:
            if len(projectionAttrs) >=1 and len(projectionMeasures) == 0:
                result = collection.distinct(projectionAttrs, filter)
            else:
                cursor = collection.aggregate(
                    mongo_api.generateAggrColumns(projectionAttrs, projectionMeasures, filter, sortSpec))

        if cursor is not None:
            for n in cursor:
                result.append(n)
        df = pd.DataFrame(data=result)
        return df

def main(argv):
    pickle_file_name = None
    collection_name = None
    query_type = None
    symbol = None
    try:
        opts, args = getopt.getopt(argv, "ht:p:c:s:", ["help", "type=", "pickle=", "mongoCollection=", "symbol="])
    except getopt.GetoptError:
        print(sys.argv[0] + ' -t type <import|export> [-p pickle_file_name] -c collection_name [-s symbol] ')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(sys.argv[0] + ' -p pickle_file_name -c collection_name')
            sys.exit()
        elif opt == '-t':
            query_type = arg
        elif opt == '-p':
            pickle_file_name = arg
        elif opt == '-c':
            collection_name = arg
        elif opt == '-s':
            symbol = arg
    if query_type == 'import' and (collection_name is None or pickle_file_name is None):
        print(sys.argv[0] + ' Make sure setting two arguments: -p pickle_file_name -c collection_name')
        sys.exit()

    m = mongo_api()
    if query_type == 'import' and os.path.exists(pickle_file_name):
        df = pd.read_pickle(pickle_file_name)
        m.write_df(df, collection_name)
    if query_type == 'export':
        distinct = False
        #projectionAttrs = ['year', 'month', 'd_index', 'symbol']
        projectionAttrs = [{'date': {'$dateFromParts': {'year': '$year', 'month': '$month', 'day': '$d_index'}}}, 'symbol']
        projectionMeasures = ['callvol', 'calloi', 'putvol', 'putoi']
        sortSpec = ['symbol', 'year', 'month', 'd_index']
        filter = {'symbol': {'$in': [symbol]}}
        sortSpec = {'symbol': 1, 'date': 1}
        df = m.read_df(collection_name, distinct, projectionAttrs, projectionMeasures, filter, sortSpec)
        print(df)
if __name__ == "__main__":
    main(sys.argv[1:])