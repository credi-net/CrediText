import requests
from datetime import datetime

def get_rdap_bootstrap():
    """Fetch the IANA RDAP bootstrap file mapping TLDs to servers."""
    response = requests.get('https://data.iana.org/rdap/dns.json')
    response.raise_for_status()
    bootstrap = response.json()
    tld_map = {}
    for service in bootstrap.get('services', []):
        tlds = service[0]
        servers = service[1]
        for tld in tlds:
            if servers:
                tld_map[tld.lower()] = servers[0]
    return tld_map

def query_domain_rdap(domain):
    """Query RDAP data for a specific domain."""
    parts = domain.lower().split('.')
    tld = parts[-1]
    tld_map = get_rdap_bootstrap()

    if tld not in tld_map:
        raise ValueError(f"No RDAP server found for .{tld} domains")

    rdap_server = tld_map[tld]
    query_url = f"{rdap_server}domain/{domain}"

    headers = {
        'Accept': 'application/rdap+json',
        'User-Agent': 'DomainMonitor/1.0'
    }

    response = requests.get(query_url, headers=headers, timeout=10)

    if response.status_code == 404:
        return {'status': 'not_found', 'domain': domain}
    elif response.status_code == 429:
        raise Exception('Rate limited')

    response.raise_for_status()
    return response.json()

def parse_rdap_response(rdap_data):
    """Extract useful fields from RDAP response."""
    if not rdap_data:
        return {'domain': None, 'status': 'available', 'expiration': None, 'registrar': None}

    domain = rdap_data.get('ldhName')
    events = rdap_data.get('events', [])
    expiration_date = None
    registration_date = None

    for event in events:
        action = event.get('eventAction')
        date_str = event.get('eventDate')
        if action == 'expiration':
            expiration_date = date_str
        elif action == 'registration':
            registration_date = date_str

    registrar_name = None
    for entity in rdap_data.get('entities', []):
        if 'registrar' in entity.get('roles', []):
            vcard = entity.get('vcardArray', [None, []])
            for field in vcard[1]:
                if field[0] == 'fn':
                    registrar_name = field[3]
                    break

    return {
        'domain': domain,
        'status': 'registered',
        'expiration': expiration_date,
        'registration': registration_date,
        'registrar': registrar_name,
        'status_codes': rdap_data.get('status', [])
    }

def get_whois_data(domain):
    # Fetch the whois information
    domain_info = whois.whois(domain)
    # Print common data fields
    # print(f"Domain Name: {domain_info.domain_name}")
    # print(f"Registrar: {domain_info.registrar}")
    # print(f"Creation Date: {domain_info.creation_date}")
    # print(f"Expiration Date: {domain_info.expiration_date}")
    # print(f"Name Servers: {domain_info.name_servers}")
    # print(domain_info)
    return domain_info


import pandas as pd
import time
from tqdm import tqdm
import whois
import json
# !pip install python-whois
if __name__ == "__main__":
    # Load the Parquet file
    df = pd.read_csv("/home/mila/a/abdallah/scratch/hsh_projects/CrediText/data/weaksupervision/domain_rel_domains.csv")
    ######### use rdap service #########
    domain_lst=list(df["domain"].unique())

    domain_rel_rdap_dict={}
    # for d in tqdm(domain_lst[0:100]):
    for idx,d in enumerate(tqdm(domain_lst)):
        try:
            d_rdap_info=query_domain_rdap(d)
            domain_rel_rdap_dict[d]=d_rdap_info
            # print(d, str(d_rdap_info)[0:50])
        except Exception as e:
            domain_rel_rdap_dict[d]="Exception: "+str(e)
            # print(d, str(e))
        time.sleep(0.05)
        if idx % 1000 == 0 and idx > 0:
            print("save idx=",idx)
            whois_info_df = pd.DataFrame(list(domain_rel_rdap_dict.items()), columns=['domain', 'whois_info'])
            whois_info_df['whois_info'] = whois_info_df['whois_info'].apply(lambda x: json.dumps(x) if isinstance(x, dict) else str(x))
            whois_info_df.to_parquet("domain_dec2024_rel_whois_info.parquet", index=False)

    rdap_exception_dict={k:v  for k,v in domain_rel_rdap_dict.items() if str(v).startswith("Exception: ")}
    # len(rdap_exception_dict) 

    whois_exception_dict={}
    for k in tqdm(rdap_exception_dict.keys()):
        try:
            whois_exception_dict[k]=get_whois_data(k)
        except Exception as e :
            whois_exception_dict[k]="Exception: "+str(e)

    for k,v in whois_exception_dict.items():
        domain_rel_rdap_dict[k]=v
    whois_info_df = pd.DataFrame(list(domain_rel_rdap_dict.items()), columns=['domain', 'whois_info'])
    whois_info_df['whois_info'] = whois_info_df['whois_info'].apply(lambda x: json.dumps(x) if isinstance(x, dict) else str(x))
    whois_info_df.to_parquet("domain_dec2024_rel_whois_info.parquet", index=False)

    # {k:v  for k,v in exec_domain_dict2.items() if str(v).startswith("Exception: ")}
    # set([elem.split(".")[-1] for elem in  set(domain_rel_rdap_dict.keys())-set(exc_dict.keys())])
    # set([elem.split(".")[-1] for elem in  set(exc_dict.keys())])