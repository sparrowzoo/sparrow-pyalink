from sparrow.xml_db.etl_template import EtlTemplate


class ETL:
    def __init__(self,xml_path):

        content= open(xml_path).readline()
        ##xml parser
        for(sql:sql_array){
            etl_template = EtlTemplate()
        }
            ## -->name
            ## -->value etl_template
        self.template_map[name]=etl_template;

    def load(self, name,args):
        ##${name}==args[0]
        sql= self.etl_template[name]
        return spark.sql(sql,args)