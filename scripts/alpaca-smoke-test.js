import { createAlpacaClient, loadDotEnv } from '../src/index.js';

loadDotEnv();

try {
  const alpaca = createAlpacaClient();
  const [clock, account, positions, orders] = await Promise.all([
    alpaca.getClock(),
    alpaca.getAccount(),
    alpaca.listPositions(),
    alpaca.listOrders({ status: 'open', limit: 50 })
  ]);

  console.log(
    JSON.stringify(
      {
        market: {
          isOpen: clock.is_open,
          nextOpen: clock.next_open,
          nextClose: clock.next_close
        },
        account: {
          status: account.status,
          currency: account.currency,
          equity: account.equity,
          cash: account.cash,
          buyingPower: account.buying_power,
          tradingBlocked: account.trading_blocked
        },
        positions: positions.length,
        openOrders: orders.length
      },
      null,
      2
    )
  );
} catch (error) {
  console.error(error.message);
  process.exitCode = 1;
}
