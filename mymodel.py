from transformers import T5ForConditionalGeneration, LlamaForCausalLM
from torch.nn import CrossEntropyLoss, NLLLoss
import torch.nn as nn
import torch
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    CausalLMOutputWithPast
)
import pdb
import torch.nn.functional as F

class MyllamaModel(LlamaForCausalLM):

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        nli_weighted: Optional[bool] = None,
        nli_weights: Optional[torch.FloatTensor] = None,
        contrast: Optional[bool] = None, 
        ref_input_ids: Optional[torch.LongTensor] = None,
        ref_attention_mask: Optional[torch.FloatTensor] = None,
        disturb_input_ids: Optional[torch.LongTensor] = None,
        disturb_attention_mask: Optional[torch.FloatTensor] = None,
        unlikelihood: Optional[bool] = None,
        batch=None,
    ):
    # ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )     
    

    def disturb_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,        
        nli_weighted: Optional[bool] = None,
        nli_weights: Optional[torch.FloatTensor] = None,
        contrast: Optional[bool] = None, 
        ref_input_ids: Optional[torch.LongTensor] = None,
        ref_attention_mask: Optional[torch.FloatTensor] = None,
        disturb_input_ids: Optional[torch.LongTensor] = None,
        disturb_attention_mask: Optional[torch.FloatTensor] = None, 
        unlikelihood: Optional[bool] = None,
        typelist: Optional[torch.LongTensor] = None,
        batch=None,
    ):
    # ) -> Union[Tuple, CausalLMOutputWithPast]:
        newtypelist = [] 
        if unlikelihood:
            for idx in range(len(typelist)):
                if typelist[idx] == 0:
                    newtypelist.append(idx) 
            if len(newtypelist) == 0:
                return None
            else:
                new_input_ids = []
                new_attn_mask = []
                new_labels = []
                for idx in newtypelist:
                    new_input_ids.append(disturb_input_ids[idx,:].tolist())
                    new_attn_mask.append(disturb_attention_mask[idx,:].tolist())
                    new_labels.append(labels[idx,:].tolist())
                new_input_ids = torch.tensor(new_input_ids).cuda()
                new_attention_mask = torch.tensor(new_attn_mask).cuda()
                new_labels = torch.tensor(new_labels).cuda()
        else:
            new_input_ids = disturb_input_ids
            new_attention_mask = disturb_attention_mask
            new_labels = labels
                
            
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=new_input_ids,
            attention_mask=new_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()

            if not unlikelihood:
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = new_labels[..., 1:].contiguous()
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
            else:
                logprobs = F.log_softmax(shift_logits, dim=-1)
                lprobs = logprobs.view(-1, logprobs.size(-1))
                one_minus_probs = torch.clamp((1 - torch.exp(lprobs)), min=1e-10)
                loss = -torch.log(one_minus_probs)
                loss = torch.mean(loss)
                

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )     
 


class MyModel(T5ForConditionalGeneration):

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        nli_weighted: Optional[bool] = None,
        nli_weights: Optional[torch.FloatTensor] = None,
        contrast: Optional[bool] = None, 
        ref_input_ids: Optional[torch.LongTensor] = None,
        ref_attention_mask: Optional[torch.FloatTensor] = None,
        disturb_input_ids: Optional[torch.LongTensor] = None,
        disturb_attention_mask: Optional[torch.FloatTensor] = None,
        unlikelihood: Optional[bool] = None,
        batch=None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:

        # print(f'forward: {input_ids.shape}', flush=True)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        batch_size = len(input_ids)
        loss = None
        if labels is not None:
            if not nli_weighted and not contrast:
                loss_fct = CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

            elif not contrast:
                # pdb.set_trace()
                loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
                loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
                loss = loss.view(batch_size, -1)

                # loss = torch.mean(loss, dim=1)
                weiloss = loss * nli_weights.unsqueeze(1)
                loss = torch.mean(weiloss)

            elif contrast:
                loss_fct = NLLLoss(ignore_index=-100, reduction='none')
                loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
                loss = -loss.view(batch_size,-1).mean(dim=1) #todo

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


    def disturb_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        nli_weighted: Optional[bool] = None,
        nli_weights: Optional[torch.FloatTensor] = None,
        contrast: Optional[bool] = None, 
        ref_input_ids: Optional[torch.LongTensor] = None,
        ref_attention_mask: Optional[torch.FloatTensor] = None,
        disturb_input_ids: Optional[torch.LongTensor] = None,
        disturb_attention_mask: Optional[torch.FloatTensor] = None, 
        unlikelihood: Optional[bool] = None,
        typelist: Optional[torch.LongTensor] = None,
        batch=None,
        ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        newtypelist = []
        # print(f'{len(typelist)}, {disturb_input_ids.shape}', flush=True)
        if unlikelihood: 
            for idx in range(len(typelist)):
                if typelist[idx] == 2:
                    newtypelist.append(idx)

            if len(newtypelist) == 0: 
                return None
            else:
                # print(disturb_input_ids.shape,flush=True)
                # print(disturb_attention_mask.shape,flush=True)
                new_input_ids = []
                new_attn_mask = []
                new_labels = []
                for idx in newtypelist:
                    new_input_ids.append(disturb_input_ids[idx,:].tolist())
                    new_attn_mask.append(disturb_attention_mask[idx,:].tolist())
                    new_labels.append(labels[idx,:].tolist())
                new_input_ids = torch.tensor(new_input_ids).cuda()
                new_attention_mask = torch.tensor(new_attn_mask).cuda()
                new_labels = torch.tensor(new_labels).cuda()
        else:
            new_input_ids = disturb_input_ids
            new_attention_mask = disturb_attention_mask
            new_labels = labels

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=new_input_ids,
                attention_mask=new_attention_mask,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,)

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if new_labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(new_labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if disturb_attention_mask is not None:
                disturb_attention_mask = disturb_attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=None,
            past_key_values=None,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=new_attention_mask,
            head_mask=None,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        batch_size = len(new_input_ids)
        loss = None
        if new_labels is not None:
            if not nli_weighted and not contrast and not unlikelihood:
                loss_fct = CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), new_labels.view(-1))

            elif nli_weighted and not contrast and not unlikelihood:
                # pdb.set_trace()
                loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
                loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), new_labels.view(-1))
                loss = loss.view(batch_size, -1)

                # loss = torch.mean(loss, dim=1)
                weiloss = loss * nli_weights.unsqueeze(1)
                loss = torch.mean(weiloss)

            elif unlikelihood and not contrast:
                logprobs = F.log_softmax(lm_logits, dim=-1)
                # logprobs = lsoftmax(lm_logits, dim=-1)
                lprobs = logprobs.view(-1, logprobs.size(-1))
                # one_minus_probs = torch.clamp((1.0 - lprobs.exp()), min=1e-10).view(-1, lprobs.view(-1))
                one_minus_probs = torch.clamp((1.0 - lprobs.exp()), min=1e-10)
                # new_labels1 = torch.where(new_labels<0, torch.zeros_like(new_labels), new_labels)
                # negative_tgts = torch.zeros_like(lprobs).scatter_(1, new_labels1.view(-1).unsqueeze(1), 1)
                # print(f'new labels {new_labels.shape}; lprobs {lprobs.shape}; negative_tgts {negative_tgts.shape}',flush=True)
                # loss = -torch.log(one_minus_probs) * negative_tgts #?
                loss = -torch.log(one_minus_probs)
                loss = torch.mean(loss)

            elif contrast:
                loss_fct = NLLLoss(ignore_index=-100, reduction='none')
                loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), new_labels.view(-1))
                # loss = -loss.view(batch_size,-1).sum(dim=1)
                loss = -loss.view(batch_size,-1).mean(dim=1)

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


    def ref_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        nli_weighted: Optional[bool] = None,
        nli_weights: Optional[torch.FloatTensor] = None,
        contrast: Optional[bool] = None, 
        ref_input_ids: Optional[torch.LongTensor] = None,
        ref_attention_mask: Optional[torch.FloatTensor] = None,) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=ref_input_ids,
                attention_mask=ref_attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if ref_attention_mask is not None:
                ref_attention_mask = ref_attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=ref_attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        batch_size = len(ref_input_ids)
        loss = None
        if labels is not None:
            if not nli_weighted and not contrast:
                loss_fct = CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

            elif not contrast:
                # pdb.set_trace()
                loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
                loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
                loss = loss.view(batch_size, -1)

                # loss = torch.mean(loss, dim=1)
                weiloss = loss * nli_weights.unsqueeze(1)
                loss = torch.mean(weiloss)

            elif contrast:
                loss_fct = NLLLoss(ignore_index=-100, reduction='none')
                loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
                # loss = -loss.view(batch_size,-1).sum(dim=1)
                loss = -loss.view(batch_size,-1).mean(dim=1)

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

