
import duckdb
import pandas as pd
from pyspark.sql.types import IntegerType, StringType, StructField, StructType,ArrayType
from regex import P
from sparkcc import CCSparkJob
# os.environ["PYSPARK_SUBMIT_ARGS"] = "--driver-memory MEM 4g"
from urllib.parse import urljoin, urlparse
from jsonpath_ng import jsonpath, parse as jsonpath_parse
import os
import shutil
import urllib
import requests
from pyspark.sql import Window
from pyspark.sql.functions import row_number
from pyspark.sql import functions as F
from pyspark.sql.functions import input_file_name
from resiliparse.extract.html2text import extract_plain_text
class ExtractwarcHTMLJob(CCSparkJob):
    """Extract links from WAT files and redirects from WARC files
    and save them as pairs <from, to>.
    """
    num_input_partitions = 64
    num_output_partitions = 4
    name = 'ExtractwarcHTMLJob'
    # include_html = False
    output_schema = StructType(
        [StructField('FileName', StringType(), True),
         StructField('Domain_Name', StringType(), True),
         StructField('WARC_Target_URI', StringType(), True),
         StructField('WARC_Identified_Content_Language', StringType(), True),
         StructField('WARC_Date', StringType(), True),
         StructField('Content_Type', StringType(), True),
         StructField('Content_Length', IntegerType(), True),
        #  StructField('warc_record_html', StringType(), True),
         StructField('warc_record_text_resili', StringType(), True),
         ]
    )
    records_response = None
    records_response_warc = None
    records_failed = None
    domains_pc1_dict=None
    allowed_offsets=None
    supported_langs = ["eng", "fra"] # "language codes: ISO-639-3 "


    def add_arguments(self, parser):
        parser.add_argument(
            '--intermediate_output',
            type=str,
            default=None,
            help='Intermediate output to recover job from',
        )

    @staticmethod
    def _url_join(base, link):
        # TODO: efficiently join without reparsing base
        # TODO: canonicalize
        pass

    def iterate_records(self, warc_uri, archive_iterator):
        count=0
        file_name=warc_uri.split("/segments/")[-1]
        warc_path=self.get_warc_file_path(warc_uri)
        allowed_offsets_dict=self.allowed_offsets.value
        if warc_path in allowed_offsets_dict:
            # print(f"Processing warc_path : {warc_path}")
            for record in archive_iterator:  
                # print(f"Processing record {record} ")            
                WARC_Identified_Content_Language = record.rec_headers['WARC-Identified-Content-Language']
                warc_record_html = self.get_payload_stream(record).read().decode('utf-8', errors='backslashreplace')                
                # warc_record_html = record.content_stream().read().decode('utf-8', errors='replace')
                offset=archive_iterator.get_record_offset()
                if offset in allowed_offsets_dict[warc_path]:
                    # print(f"Processing record with offset : {offset}")
                    for res in self.process_record(record, warc_record_html,WARC_Identified_Content_Language):
                        Domain_Name, WARC_Target_URI, WARC_Identified_Content_Language, WARC_Date, Content_Type,Content_Length,warc_record_text_resili=res #warc_record_html
                        if Domain_Name:
                            yield (file_name,)+res
                self.records_processed.add(1)


    @staticmethod
    def is_domain_exist(domain_name:str, url = "http://206.12.91.10:22101/searchDomainName/"):
        data = {"domainName": domain_name}
        response = requests.post(url, json=data)
        # print(f"Status Code: {response.status_code}")
        return False if response.json()['domain_exist']==0 else True

    def process_record(self, record, warc_record_html,WARC_Identified_Content_Language):
        self.records_response.add(1)
        # print(f"record={record.rec_headers}")
        if (record.rec_headers["WARC-Type"] == 'response' and
                record.rec_headers["WARC-Identified-Payload-Type"] == "text/html"):
            self.records_response_warc.add(1)           
            Domain_Name, WARC_Target_URI, WARC_Date, Content_Type, Content_Length, warc_record_text_resili = \
                None, None, None, None, None, None
            if 1 == 1:
                WARC_Target_URI = record.rec_headers['WARC-Target-URI']
                Domain_Name = urllib.parse.unquote(
                    WARC_Target_URI.split("/")[2].split("www.")[-1].split("?")[0]
                )
                # if Domain_Name in self.domains_set.value:
                WARC_Date      = record.rec_headers['WARC-Date']
                Content_Type   = record.rec_headers['Content-Type']
                Content_Length = int(record.rec_headers['Content-Length'])
                # warc_record_html = self.get_payload_stream(record).read().decode('utf-8', errors='backslashreplace')
                # print(f"warc_record_html {warc_record_html[:100]}")
                warc_record_text_resili = extract_plain_text(warc_record_html, main_content=True)                

            yield (Domain_Name, WARC_Target_URI, WARC_Identified_Content_Language, WARC_Date,
                Content_Type, Content_Length, warc_record_text_resili) #warc_record_html
            return (None, None, None, None,  None, None, None)

        else:
            return (None, None, None, None,  None, None, None)
    def process_record_without_offset(self, record):
        # print(f"record={record.rec_headers}")
        self.records_response.add(1)
        if  record.rec_headers["WARC-Type"] == 'response' and record.rec_headers["WARC-Identified-Payload-Type"]=="text/html":
            # print(f"Processing HTML record: {record.rec_headers['WARC-Target-URI']}")
            self.records_response_warc.add(1)
            WARC_Identified_Content_Language = record.rec_headers['WARC-Identified-Content-Language']
            if WARC_Identified_Content_Language:
                WARC_Identified_Content_Languages_lst = WARC_Identified_Content_Language.split(",")
            Domain_Name, WARC_Target_URI, WARC_Date, Content_Type, Content_Length, warc_record_html,warc_record_text_resili = None, None, None, None, None, None,None
            # if len(set(WARC_Identified_Content_Languages_lst) & set(self.supported_langs)) > 0:
            if 1 == 1:
                WARC_Target_URI = record.rec_headers['WARC-Target-URI']
                Domain_Name = urllib.parse.unquote(WARC_Target_URI.split("/")[2].split("www.")[-1].split("?")[0])
                # print(f" Domain_Name: {Domain_Name} ")
                if Domain_Name in self.domains_set.value:
                    # if self.is_domain_exist(Domain_Name):
                    # print(f"{Domain_Name} exist")
                    WARC_Date = record.rec_headers['WARC-Date']
                    Content_Type = record.rec_headers['Content-Type']
                    Content_Length = int(record.rec_headers['Content-Length'])
                    warc_record_html = self.get_payload_stream(record).read().decode('utf-8',errors='backslashreplace') # handel invalid encoding by replacing with backslash escape sequence
                    warc_record_text_resili=extract_plain_text(warc_record_html,main_content=True)
                else:
                    # print(f"{Domain_Name} Not exist")
                    Domain_Name = None
            yield (Domain_Name, WARC_Target_URI, WARC_Identified_Content_Language, WARC_Date, Content_Type,
                    Content_Length, warc_record_html,warc_record_text_resili)
            return (None, None, None, None, None, None, None,None)
        else:
            return (None, None, None, None, None, None, None,None)

    def init_accumulators(self, session):
        super(ExtractwarcHTMLJob, self).init_accumulators(session)
        sc = session.sparkContext
        self.records_failed = sc.accumulator(0)
        # self.records_non_html = sc.accumulator(0)
        self.records_response = sc.accumulator(0)
        self.records_response_warc = sc.accumulator(0)
        # self.records_response_redirect = sc.accumulator(0)
        # self.records_response_robotstxt = sc.accumulator(0)
        # self.link_count = sc.accumulator(0)

    def log_accumulators(self, session):
        super(ExtractwarcHTMLJob, self).log_accumulators(session)

        self.log_accumulator(session, self.records_response, 'response records = {}')
        self.log_accumulator(
            session, self.records_failed, 'records failed to process = {}'
        )
        # self.log_accumulator(session, self.records_non_html, 'records not HTML = {}')
        self.log_accumulator(
            session, self.records_response_warc, 'response records warc = {}'
        )

        # self.log_accumulator(
        #     session, self.records_response_redirect, 'response records redirects = {}'
        # )
    @staticmethod
    def load_domain_pc1(domains_pc1_csv_path="../../data/dqr/domain_pc1.csv"):
        doamins_df = pd.read_csv(domains_pc1_csv_path)
        return dict(zip(doamins_df["domain"].tolist(), doamins_df["pc1"].tolist()))
    @staticmethod
    def query_parquet_duckdb(SQL_Query:str,max_memory:str="4GB",threads:int =8,batch_size:int =int(1e4)):
        con = duckdb.connect()
        con.execute(f"SET memory_limit='{max_memory}'")
        con.execute(f"SET threads={threads}")
        try:
            result = con.execute(SQL_Query)
        except Exception as e:
            print(f"Error occurred while executing SQL query:{e}\nSQL_Query={SQL_Query}")
            raise

        cols_map={name[0]:idx for idx,name in enumerate(result.description)}
        res=[]
        while True:
            rows = result.fetchmany(int(batch_size))
            if not rows:
                break
            for row in rows:
                res.append([elem for elem in row])
        res=pd.DataFrame(res,columns=cols_map.keys())
        return res

    @staticmethod
    def get_warc_file_path(full_path:str):
        return "crawl-data/"+str(full_path).split(":")[-1].split("crawl-data/")[-1]
    def run_job(self, session):
        self.get_logger().info(f"seed domain path={self.args.trusted_domains}")
        out_path=str(session.conf.get("spark.sql.warehouse.dir")).split(":")[-1]+"/"+self.args.output
        if self.args.input != '':
            input_data = session.sparkContext.textFile(
            # input_data=session.sparkContext.wholeTextFiles(
                self.args.input, minPartitions=self.args.num_input_partitions
            )
            output = input_data.mapPartitionsWithIndex(self.process_warcs)

        warc_files_in_job = set(input_data.collect())        
        warc_files_in_job=set( [self.get_warc_file_path(elem) for elem in warc_files_in_job] )
        print(f"warc_files_in_job={len(warc_files_in_job)}")

        # file_offset_df=pd.read_parquet(self.args.trusted_domains)
        SQL_Query=f"SELECT warc_filename,warc_record_offset FROM read_parquet('{self.args.trusted_domains}') where warc_filename in {tuple(warc_files_in_job)}"
        print(f"Offset SQL_Query={SQL_Query}")
        file_offset_df=ExtractwarcHTMLJob.query_parquet_duckdb(SQL_Query=SQL_Query)
        print(f"len of file_offset_df={len(file_offset_df)}")

        
        file_offset_df=file_offset_df[file_offset_df["warc_filename"].isin(warc_files_in_job)]
        allowed_offsets_set = set(zip(file_offset_df['warc_filename'], file_offset_df['warc_record_offset']))
        allowed_offsets_dict = {}
        for warc_filename, warc_record_offset in allowed_offsets_set:
            if warc_filename not in allowed_offsets_dict:
                allowed_offsets_dict[warc_filename] = []
            allowed_offsets_dict[warc_filename].append(warc_record_offset)

        # print(f"allowed_offsets_set={allowed_offsets_dict}")
        print(f"len(allowed_offsets_set)={len(allowed_offsets_set)}")
        self.allowed_offsets=session.sparkContext.broadcast(allowed_offsets_dict)
        del allowed_offsets_set
        del allowed_offsets_dict
        
        # cc_label_deg_3_df=pd.read_csv(self.args.trusted_domains)        
        # cc_label_deg_3_set=set(cc_label_deg_3_df["domain"].tolist())  
        # print(f"len(domains_set)={len(cc_label_deg_3_set)}")
        # del cc_label_deg_3_df
        # self.domains_set = session.sparkContext.broadcast(cc_label_deg_3_set)        
        # del cc_label_deg_3_set
        # self.domains_pc1_dict = session.sparkContext.broadcast(self.load_domain_pc1(self.args.trusted_domains))

        if os.path.exists(out_path):
            shutil.rmtree(out_path)
        
        if not self.args.intermediate_output:
            df = session.createDataFrame(output, schema=self.output_schema)
        else:
            if output is not None:
                session.createDataFrame(output, schema=self.output_schema).write.format(
                    self.args.output_format
                ).option('compression', self.args.output_compression).saveAsTable(
                    self.args.intermediate_output
                )
                self.log_accumulators(session.sparkContext)
            warehouse_dir = session.conf.get(
                'spark.sql.warehouse.dir', 'spark-warehouse'
            )
            intermediate_output = os.path.join(
                warehouse_dir, self.args.intermediate_output
            )
            df = session.read.parquet(intermediate_output)
        df=df.dropDuplicates()
        df_final=df.orderBy(F.col("Domain_Name").asc(), F.col("Content_Length").asc())
        ################## filter by 3 min-max records per domain ##################
        # df = df.filter(df.Content_Length >= 500)
        # w_desc = Window.partitionBy("Domain_Name").orderBy(F.col("Content_Length").desc())
        # w_asc = Window.partitionBy("Domain_Name").orderBy(F.col("Content_Length").asc())
        # df_low = (df
        #           .withColumn("rn", F.row_number().over(w_asc))
        #           .filter(F.col("rn") <= 2)
        #           .drop("rn")
        #           )
        # df_high = (df
        #            .withColumn("rn", F.row_number().over(w_desc))
        #            .filter(F.col("rn") <= 2)
        #            .drop("rn")
        #            )
        # df_final = df_low.union(df_high).distinct()
        # df_final = df_final.orderBy(F.col("Domain_Name").asc(), F.col("Content_Length").asc())
        # df_sorted.collect()
        # df_sorted_Domain_Name_counts = df_final.groupBy("Domain_Name").count().collect()
        #######################################        
        df_final.coalesce(self.args.num_output_partitions).write.format(self.args.output_format).option(
            'compression', self.args.output_compression
        ).mode("overwrite").saveAsTable(self.args.output)

        self.log_accumulators(session.sparkContext)




if __name__ == '__main__':
    job = ExtractwarcHTMLJob()
    # ExtractwarcContentsJob.domains_pc1_dict=load_domain_pc1()
    job.run()


