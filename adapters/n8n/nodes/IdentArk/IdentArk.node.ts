import {
	IExecuteFunctions,
	INodeExecutionData,
	INodeType,
	INodeTypeDescription,
	NodeOperationError,
} from 'n8n-workflow';

export class IdentArk implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'IdentArk',
		name: 'identark',
		icon: 'file:identark.svg',
		group: ['transform'],
		version: 1,
		subtitle: '={{$parameter["operation"]}}',
		description: 'Secure AI agent gateway — invoke LLMs without exposing credentials',
		defaults: {
			name: 'IdentArk',
		},
		inputs: ['main'],
		outputs: ['main'],
		credentials: [
			{
				name: 'identarkApi',
				required: true,
			},
		],
		properties: [
			{
				displayName: 'Operation',
				name: 'operation',
				type: 'options',
				noDataExpression: true,
				options: [
					{
						name: 'Invoke LLM',
						value: 'invokeLlm',
						description: 'Send messages to an LLM and get a response',
						action: 'Invoke LLM',
					},
					{
						name: 'Create Session',
						value: 'createSession',
						description: 'Create a new agent session with specific configuration',
						action: 'Create session',
					},
					{
						name: 'Get Session Cost',
						value: 'getSessionCost',
						description: 'Get the current cost for a session',
						action: 'Get session cost',
					},
				],
				default: 'invokeLlm',
			},

			// ── Invoke LLM fields ─────────────────────────────────────────────────
			{
				displayName: 'Message',
				name: 'message',
				type: 'string',
				typeOptions: {
					rows: 4,
				},
				default: '',
				required: true,
				displayOptions: {
					show: {
						operation: ['invokeLlm'],
					},
				},
				description: 'The message to send to the LLM',
			},
			{
				displayName: 'Session ID',
				name: 'sessionId',
				type: 'string',
				default: '',
				displayOptions: {
					show: {
						operation: ['invokeLlm', 'getSessionCost'],
					},
				},
				description: 'Session ID to continue a conversation (optional for invokeLlm)',
			},
			{
				displayName: 'Role',
				name: 'role',
				type: 'options',
				options: [
					{ name: 'User', value: 'user' },
					{ name: 'System', value: 'system' },
					{ name: 'Assistant', value: 'assistant' },
				],
				default: 'user',
				displayOptions: {
					show: {
						operation: ['invokeLlm'],
					},
				},
				description: 'The role of the message sender',
			},

			// ── Create Session fields ─────────────────────────────────────────────
			{
				displayName: 'Agent ID',
				name: 'agentId',
				type: 'string',
				default: '',
				required: true,
				displayOptions: {
					show: {
						operation: ['createSession'],
					},
				},
				description: 'Identifier for this agent',
			},
			{
				displayName: 'Model',
				name: 'model',
				type: 'options',
				options: [
					{ name: 'GPT-4o', value: 'gpt-4o' },
					{ name: 'GPT-4o Mini', value: 'gpt-4o-mini' },
					{ name: 'Claude Sonnet 4', value: 'claude-sonnet-4-6' },
					{ name: 'Claude Haiku', value: 'claude-haiku-4-5-20251001' },
					{ name: 'Mistral Large', value: 'mistral-large-latest' },
					{ name: 'Mistral Small', value: 'mistral-small-latest' },
				],
				default: 'gpt-4o',
				displayOptions: {
					show: {
						operation: ['createSession'],
					},
				},
				description: 'The LLM model to use',
			},
			{
				displayName: 'Provider',
				name: 'provider',
				type: 'options',
				options: [
					{ name: 'OpenAI', value: 'openai' },
					{ name: 'Anthropic', value: 'anthropic' },
					{ name: 'Mistral AI (EU)', value: 'mistral' },
					{ name: 'Azure OpenAI (UK)', value: 'azure_openai' },
					{ name: 'AWS Bedrock (UK)', value: 'bedrock' },
				],
				default: 'openai',
				displayOptions: {
					show: {
						operation: ['createSession'],
					},
				},
				description: 'The LLM provider',
			},
			{
				displayName: 'Credential Reference',
				name: 'credentialRef',
				type: 'string',
				default: '',
				required: true,
				displayOptions: {
					show: {
						operation: ['createSession'],
					},
				},
				description: 'Vault path to the LLM credential (e.g., secret/orgs/{org}/providers/openai)',
			},
			{
				displayName: 'Cost Cap (USD)',
				name: 'costCapUsd',
				type: 'number',
				default: 5.0,
				displayOptions: {
					show: {
						operation: ['createSession'],
					},
				},
				description: 'Maximum cost allowed for this session',
			},
		],
	};

	async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
		const items = this.getInputData();
		const returnData: INodeExecutionData[] = [];
		const operation = this.getNodeParameter('operation', 0) as string;
		const credentials = await this.getCredentials('identarkApi');
		const baseUrl = credentials.baseUrl as string;

		for (let i = 0; i < items.length; i++) {
			try {
				let responseData: object;

				if (operation === 'invokeLlm') {
					const message = this.getNodeParameter('message', i) as string;
					const role = this.getNodeParameter('role', i) as string;
					const sessionId = this.getNodeParameter('sessionId', i) as string;

					const body: Record<string, unknown> = {
						new_messages: [{ role, content: message }],
					};
					if (sessionId) {
						body.session_id = sessionId;
					}

					responseData = await this.helpers.httpRequestWithAuthentication.call(
						this,
						'identarkApi',
						{
							method: 'POST',
							url: `${baseUrl}/v1/llm/invoke`,
							body,
							json: true,
						},
					);
				} else if (operation === 'createSession') {
					const agentId = this.getNodeParameter('agentId', i) as string;
					const model = this.getNodeParameter('model', i) as string;
					const provider = this.getNodeParameter('provider', i) as string;
					const credentialRef = this.getNodeParameter('credentialRef', i) as string;
					const costCapUsd = this.getNodeParameter('costCapUsd', i) as number;

					responseData = await this.helpers.httpRequestWithAuthentication.call(
						this,
						'identarkApi',
						{
							method: 'POST',
							url: `${baseUrl}/v1/sessions`,
							body: {
								agent_id: agentId,
								model,
								provider,
								credential_ref: credentialRef,
								cost_cap_usd: costCapUsd,
							},
							json: true,
						},
					);
				} else if (operation === 'getSessionCost') {
					const sessionId = this.getNodeParameter('sessionId', i) as string;

					if (!sessionId) {
						throw new NodeOperationError(
							this.getNode(),
							'Session ID is required for Get Session Cost',
						);
					}

					responseData = await this.helpers.httpRequestWithAuthentication.call(
						this,
						'identarkApi',
						{
							method: 'GET',
							url: `${baseUrl}/v1/sessions/cost`,
							qs: { session_id: sessionId },
							json: true,
						},
					);
				} else {
					throw new NodeOperationError(this.getNode(), `Unknown operation: ${operation}`);
				}

				returnData.push({ json: responseData });
			} catch (error) {
				if (this.continueOnFail()) {
					returnData.push({
						json: { error: (error as Error).message },
						pairedItem: { item: i },
					});
					continue;
				}
				throw error;
			}
		}

		return [returnData];
	}
}
