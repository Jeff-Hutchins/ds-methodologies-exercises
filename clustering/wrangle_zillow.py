import pandas as pd
import env


def get_db_url(db, user= env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'


def get_zillow_data():
    query = '''
        select 
        svi.`COUNTY` county,
        p.`taxamount`/p.`taxvaluedollarcnt` tax_rate,
        p.`id`,
        p.`parcelid`,
        p.`airconditioningtypeid`,
        act.`airconditioningdesc`,
        p.`architecturalstyletypeid`,
        ast.`architecturalstyledesc`,
        p.`basementsqft`,
        p.`bathroomcnt`,
        p.`bedroomcnt`,
        p.`buildingclasstypeid`,
        bct.`buildingclassdesc`,
        p.`buildingqualitytypeid`,
        p.`calculatedbathnbr`,
        p.`calculatedfinishedsquarefeet`,
        p.`decktypeid`,
        p.`finishedfloor1squarefeet`,
        p.`finishedsquarefeet12`,
        p.`finishedsquarefeet13`,
        p.`finishedsquarefeet15`,
        p.`finishedsquarefeet50`,
        p.`finishedsquarefeet6`,
        p.`fips`,
        svi.`ST_ABBR` state,
        p.`fireplacecnt`,
        p.`fullbathcnt`,
        p.`garagecarcnt`,
        p.`garagetotalsqft`,
        p.`hashottuborspa`,
        p.`heatingorsystemtypeid`,
        hst.`heatingorsystemdesc`,
        p.`latitude`,
        p.`longitude`,
        p.`lotsizesquarefeet`,
        p.`poolcnt`,
        p.`poolsizesum`,
        p.`pooltypeid10`,
        p.`pooltypeid2`,
        p.`pooltypeid7`,
        p.`propertycountylandusecode`,
        p.`propertylandusetypeid`,
        plut.`propertylandusedesc`,
        p.`propertyzoningdesc`,
        p.`rawcensustractandblock`,
        p.`regionidcity`,
        p.`regionidcounty`,
        p.`regionidneighborhood`,
        p.`regionidzip`,
        p.`roomcnt`,
        p.`storytypeid`,
        st.`storydesc`,
        p.`taxvaluedollarcnt`,
        p.`threequarterbathnbr`,
        p.`unitcnt`,
        p.`yardbuildingsqft17`,
        p.`yardbuildingsqft26`,
        p.`yearbuilt`,
        p.`numberofstories`,
        p.`fireplaceflag`,
        p.`structuretaxvaluedollarcnt`,
        p.`assessmentyear`,
        p.`landtaxvaluedollarcnt`,
        p.`taxamount`,
        p.`taxdelinquencyflag`,
        p.`taxdelinquencyyear`, 
        p.`typeconstructiontypeid`,
        tct.`typeconstructiondesc`,
        p.`censustractandblock`,
        pred.`transactiondate`,
        pred.`logerror`,
        m.`transactions`
    from 
        `properties_2017` p
    inner join `predictions_2017`  pred
        on p.`parcelid` = pred.`parcelid` 
    inner join 
        (select 
            `parcelid`, 
            max(`transactiondate`) `lasttransactiondate`, 
            max(`id`) `maxid`, 
            count(*) `transactions`
        from 
            predictions_2017
        group by 
            `parcelid`
        ) m
        on 
        pred.parcelid = m.parcelid
        and pred.transactiondate = m.lasttransactiondate
    left join `propertylandusetype` plut
        on p.`propertylandusetypeid` = plut.`propertylandusetypeid`
            
    left join svi_db.svi2016_us_county svi
        on p.`fips` = svi.`FIPS`
    left join `airconditioningtype` act
        using(`airconditioningtypeid`)
    left join heatingorsystemtype hst
        using(`heatingorsystemtypeid`)
    left join `architecturalstyletype` ast
        using(`architecturalstyletypeid`)
    left join `buildingclasstype` bct
        using(`buildingclasstypeid`)
    left join `storytype` st
        using(`storytypeid`)
    left join `typeconstructiontype` tct
        using(`typeconstructiontypeid`)
    where 
        p.`latitude` is not null
        and p.`longitude` is not null;
        '''

    df = pd.read_sql(query, get_db_url('zillow'))
    return df

#Functions of the work above needed to aquire and prepare
#a new sample of data
# from aquire_zillow import get_zillow_data
# from prepare_zillow import zillow_single_unit
# from prepare_zillow import remove_columns
# from prepare_zillow import handle_missing_values
# from prepare_zillow import fill_missing_values

# def wrangle_zillow_data():
#     df=get_zillow_data()
#     df=zillow_single_unit(df)
#     df=remove_columns(df,['calculatedbathnbr','finishedsquarefeet12','fullbathcnt','propertycountylandusecode','unitcnt','structuretaxvaluedollarcnt','landtaxvaluedollarcnt','assessmentyear','propertyzoningdesc'])
#     df=handle_missing_values(df)
#     df.dropna(inplace=True)
#     return df