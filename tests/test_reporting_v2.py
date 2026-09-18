import json

import numpy as np
import pandas as pd

from nvda_quant_model.research.reporting import _dist_selection, _equity, build_report


def test_equity_and_drawdown_use_saved_net_returns_including_entry_cost():
    import matplotlib.pyplot as plt
    returns=pd.DataFrame({'Date':['2025-01-02','2025-01-03'],'rule_calibrated':[-.01,.02]})
    fig,axes=plt.subplots(1,2)
    _equity(axes[0],returns)
    _equity(axes[1],returns,drawdown=True)
    np.testing.assert_allclose(axes[0].lines[0].get_ydata(),[99,100.98])
    np.testing.assert_allclose(axes[1].lines[0].get_ydata(),[-.01,0])
    plt.close(fig)


def test_coverage_panel_never_substitutes_overlapping_windows():
    frame=pd.DataFrame([{'model':'normal','design':'overlap_supplement','horizon':21,'level':.95,'coverage':.9}])
    assert _dist_selection(frame).empty


def test_partial_run_report_exports_synthetic_watermark_and_no_fabricated_scores(tmp_path):
    (tmp_path/'manifest.json').write_text(json.dumps({'synthetic':True,'status':'running',
        'data_first':'2025-01-02','data_last':'2025-12-31','data_rows':250,
        'holdout_status':'synthetic_correctness_example_not_investment_evidence'}))
    pd.DataFrame([{'model':'logistic','mapping':mapping,'brier':value,'log_loss':.7,'n':12}
        for mapping,value in [('raw',.29),('calibrated',.28),('base_rate',.25)]]).to_csv(tmp_path/'probability_scores.csv',index=False)
    result=build_report(tmp_path)
    assert result['report']==str(tmp_path/'research_report.md')
    report=(tmp_path/'research_report.md').read_text()
    assert 'Synthetic correctness example' in report
    assert '0.2800' in report and 'did not beat' in report
    assert 'No recorded rows' in report
    for extension in ['png','pdf','svg']:
        assert (tmp_path/f'overview.{extension}').stat().st_size>1000
        assert (tmp_path/f'linkedin_main.{extension}').stat().st_size>1000
    svg=(tmp_path/'linkedin_main.svg').read_text()
    assert 'SYNTHETIC EXAMPLE' in svg and '2025-12-31' in svg
    assert (tmp_path/'figures'/'distribution_coverage_score.png').exists()


def test_cost_sensitivity_is_incremental_percentage_points_not_absolute_return():
    import matplotlib.pyplot as plt
    from nvda_quant_model.research.reporting import _cost
    data=pd.DataFrame({'model':['buy_hold']*3,'cost_multiplier':[1,2,4],
                       'total_return':[8.,7.98,7.95]})
    fig,ax=plt.subplots()
    _cost(ax,data)
    np.testing.assert_allclose(ax.lines[0].get_ydata(),[0,-2,-5])
    plt.close(fig)


def test_log_wealth_rejects_nonpositive_saved_wealth():
    import matplotlib.pyplot as plt
    import pytest
    fig,ax=plt.subplots()
    with pytest.raises(ValueError,match='positive saved wealth'):
        _equity(ax,pd.DataFrame({'Date':['2025-01-02'],'buy_hold':[-1.]}))
    plt.close(fig)
