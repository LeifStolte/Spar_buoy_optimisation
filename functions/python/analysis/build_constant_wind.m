function wind = build_constant_wind(windInput, timeInfo)

t = (0:timeInfo.dt:(timeInfo.TDur - timeInfo.dt)).';

wind.t = t;
wind.V_10 = windInput.V_10;
wind.V_hub = windInput.V_10 * ones(size(t));

end