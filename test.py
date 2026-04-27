import asyncio

async def pizza_shop():
    # 1. Create a Future (The Buzzer)
    loop = asyncio.get_running_loop()
    buzzer = loop.create_future()

    # 2. Start the "cooking" process in the background
    loop.call_later(2, lambda: buzzer.set_result("🍕 Large Pepperoni"))
    
    print("Waiting for pizza...")
    # 3. 'await' waits until someone calls set_result()
    result = await buzzer 
    print(f"Buzzer went off! Got: {result}")

asyncio.run(pizza_shop())