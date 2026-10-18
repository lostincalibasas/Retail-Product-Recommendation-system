import numpy as np
import pandas as pd
import pytest

from retail_recs.contracts import DataConfig, FEATURE_COLUMNS
from retail_recs.heldout import test_snapshot as make_snapshot


def fixture():
    tx = pd.DataFrame([(1,10,624),(1,10,625),(1,20,631),(1,30,632),(2,20,600),(99,40,625)],
                      columns=["household_key","PRODUCT_ID","DAY"])
    tx["QUANTITY"]=1.; tx["SALES_VALUE"]=2.; tx["BASKET_ID"]=np.arange(len(tx))
    products=pd.DataFrame({"PRODUCT_ID":[10,20,30,40],"COMMODITY_DESC":["A"]*4,"BRAND":["N"]*4})
    return tx,products


def test_heldout_boundaries_novelty_and_missing_future_candidates():
    tx, products=fixture()
    rows,targets,groups,summary=make_snapshot(tx,products,624,DataConfig())
    assert set(rows.PRODUCT_ID)=={10,20}  # no future-only SKU
    assert targets.PRODUCT_ID.tolist()==[10,20]  # 625 and 631 included, 632 excluded
    assert targets.repeat_product.tolist()==[True,False]
    assert set(rows.household_key)=={1,2} and summary["new_households_excluded"]==1
    assert groups.target_count.tolist()==[2,0]
    assert groups.repeat_target_count.tolist()==[1,0]
    assert groups.new_target_count.tolist()==[1,0]
    assert groups.row_offset.tolist()==[0,2]
    assert all(rows[f].dtype==np.dtype('float32') for f in FEATURE_COLUMNS)


def test_future_perturbation_preserves_test_features_and_groups():
    tx,products=fixture()
    a,_,_,_=make_snapshot(tx,products,624,DataConfig())
    tx.loc[tx.DAY>624,"QUANTITY"] = 100
    tx.loc[tx.DAY>624,"SALES_VALUE"] = 9999
    b,_,_,_=make_snapshot(tx,products,624,DataConfig())
    pd.testing.assert_frame_equal(a,b,check_exact=True)
    with pytest.raises(ValueError,match="outside"):
        make_snapshot(tx,products,617,DataConfig())


def test_later_cutoff_can_use_past_test_purchases():
    tx,products=fixture()
    rows,targets,_,_=make_snapshot(tx,products,631,DataConfig())
    assert 40 in rows.PRODUCT_ID.values and 99 in rows.household_key.values
    assert targets.PRODUCT_ID.tolist()==[30]
    assert not targets.repeat_product.any()
